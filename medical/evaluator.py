from typing import Dict, List, Any
import re


class DiagnosisEvaluator:
    """Evalúa si los diagnósticos propuestos están soportados por la base de conocimiento
    y genera una métrica (%) y una justificación textual.

    La métrica se basa en:
      - proporción de documentos recuperados que mencionan la enfermedad propuesta (db_support_ratio)
      - solapamiento de síntomas entre el caso y los documentos recuperados (symptom_support)

    Métrica final = round((db_support_ratio * db_weight + symptom_support * symptom_weight) * 100)
    """

    def __init__(self, top_k: int = 5, symptom_weight: float = 0.3, db_weight: float = 0.7):
        self.top_k = top_k
        self.symptom_weight = symptom_weight
        self.db_weight = db_weight

    def _retrieve_docs(self, sistema, sintomas_text: str) -> List[Any]:
        """Intenta recuperar documentos usando varios métodos de fallback."""
        docs = []

        if getattr(sistema, 'retriever', None) is not None:
            retr = sistema.retriever
            try:
                docs = retr.get_relevant_documents(sintomas_text)
            except Exception:
                try:
                    docs = retr.get_documents(sintomas_text)
                except Exception:
                    docs = []

        # if still empty, try vectorstore similarity search
        if not docs and getattr(sistema, 'vectorstore', None) is not None:
            try:
                docs = sistema.vectorstore.similarity_search(sintomas_text, k=self.top_k)
            except Exception:
                try:
                    docs = sistema.vectorstore.similarity_search(sintomas_text, k=max(self.top_k, 20))
                except Exception:
                    docs = []

        return docs

    def evaluate(self, resultado: Dict, caso: Dict, sistema) -> Dict:
        sintomas = caso.get('paciente', {}).get('sintomas', [])
        sintomas_text = ", ".join(sintomas)

        docs = self._retrieve_docs(sistema, sintomas_text)

        # Si no encontramos docs, devolvemos metric=None (DB no usable para este caso)
        if not docs:
            justification = (
                "No se recuperaron documentos relevantes desde la base de conocimiento para estos síntomas. "
                "Comprueba que la base está inicializada o prueba con una descripción más detallada de los síntomas."
            )
            return {
                'metric': None,
                'justification': justification,
                'supported_candidates': [],
                'hallucinated_candidates': resultado.get('diagnosticos_candidatos', [])
            }

        # Preprocess docs text
        docs_texts = [getattr(d, 'page_content', str(d)) for d in docs]
        combined = "\n\n".join(docs_texts).lower()

        supported = []
        hallucinated = []
        details = []

        candidatos = resultado.get('diagnosticos_candidatos', []) or []
        # Try to get total number of documents in the entire vectorstore (best-effort)
        total_docs = None
        if getattr(sistema, 'vectorstore', None) is not None:
            vs = sistema.vectorstore
            # multiple fallbacks to obtain total count
            try:
                if hasattr(vs, '_collection'):
                    try:
                        total_docs = vs._collection.count()
                    except Exception:
                        try:
                            total_docs = len(vs._collection.get().get('ids', []))
                        except Exception:
                            total_docs = None
            except Exception:
                total_docs = None

        # fallback: try a wide similarity_search to approximate total size
        if total_docs is None and getattr(sistema, 'vectorstore', None) is not None:
            try:
                all_docs = sistema.vectorstore.similarity_search("", k=10000)
                total_docs = len(all_docs)
            except Exception:
                try:
                    all_docs = sistema.vectorstore.similarity_search("the", k=10000)
                    total_docs = len(all_docs)
                except Exception:
                    total_docs = len(docs)

        for cand in candidatos:
            name = str(cand.get('enfermedad', '')).strip()
            name_l = name.lower()
            evid = str(cand.get('evidencia', '') or '').strip()

            # Find docs in the DB that mention the disease (attempt wide search)
            disease_docs = []
            if getattr(sistema, 'vectorstore', None) is not None:
                try:
                    disease_docs = sistema.vectorstore.similarity_search(name_l, k=10000)
                except Exception:
                    # fallback to a smaller search
                    try:
                        disease_docs = sistema.vectorstore.similarity_search(name_l, k=50)
                    except Exception:
                        disease_docs = []

            # Count how many of those disease_docs also match the symptoms (require all symptoms to appear)
            matched_count = 0
            for d in disease_docs:
                text = getattr(d, 'page_content', str(d)).lower()
                # check if all symptoms are present (strict) or at least majority
                if sintomas:
                    matches = sum(1 for s in sintomas if s.lower() in text)
                    # consider a match if >= 60% of symptoms appear
                    if matches / max(1, len(sintomas)) >= 0.6:
                        matched_count += 1
                else:
                    # if no symptoms provided, count all disease docs
                    matched_count += 1

            # db ratio based on matched_count relative to total_docs (prefer total_docs if available)
            if total_docs and total_docs > 0:
                db_ratio = matched_count / total_docs
            else:
                # fallback to ratio over disease_docs found
                db_ratio = (matched_count / len(disease_docs)) if disease_docs else 0.0

            # Symptom support overall in retrieved context
            symptom_matches = 0
            for s in sintomas:
                if s.lower() in combined:
                    symptom_matches += 1
            symptom_support = (symptom_matches / len(sintomas)) if sintomas else 0.0

            # Combine: give primary weight to db_ratio (cases-based) and minor weight to symptom overlap
            combined_score = db_ratio * 0.85 + symptom_support * 0.15

            # exact name presence boost (small), based on fraction of docs with exact name
            exact_name_count = 0
            for text in docs_texts:
                if name_l and name_l in text.lower():
                    exact_name_count += 1
            name_token_ratio = (exact_name_count / len(docs)) if docs else 0.0
            combined_score = min(1.0, combined_score + 0.05 * name_token_ratio)

            # Penalize if age/gender in the case don't appear in retrieved docs (they reduce confidence)
            try:
                case_age = int(caso.get('paciente', {}).get('edad'))
            except Exception:
                case_age = None
            case_gender = str(caso.get('paciente', {}).get('genero', '')).lower()

            age_penalty = 0.0
            gender_penalty = 0.0
            if case_age is not None:
                # if the exact age string does not appear in any doc, apply small penalty
                if not any(str(case_age) in t.lower() for t in docs_texts):
                    age_penalty = 0.04
            if case_gender and case_gender != '?':
                if not any(case_gender in t.lower() for t in docs_texts):
                    gender_penalty = 0.03

            combined_score = max(0.0, combined_score - age_penalty - gender_penalty)

            # Apply a conservative dampening so probability never reaches 100%.
            # epsilon decreases as matched_count grows: more matching cases -> smaller epsilon
            epsilon = 1.0 / (max(1, matched_count) + 10)
            adjusted_prob = combined_score * (1 - epsilon)

            # final percent and cap below 100
            pct = int(round(adjusted_prob * 100))
            if pct >= 100:
                pct = 99

            detail = {
                'enfermedad': name,
                'matched_cases': matched_count,
                'total_docs': total_docs if total_docs is not None else len(docs),
                'db_ratio': round(db_ratio, 4),
                'symptom_support': round(symptom_support, 3),
                'score_pct': pct
            }
            details.append(detail)

            if matched_count > 0:
                supported.append(cand)
            else:
                hallucinated.append(cand)

        # Metric: take the top candidate's percentage if available
        top_pct = None
        if candidatos:
            top_name = str(resultado.get('diagnostico_final', {}).get('enfermedad', '')).strip()
            top_detail = next((d for d in details if d['enfermedad'] == top_name), None)
            if top_detail:
                top_pct = top_detail['score_pct']
            else:
                top_pct = int(round(sum(d['score_pct'] for d in details)/len(details))) if details else 0

        # Build justification
        justification_lines = [f"Evaluación de {len(candidatos)} candidato(s), documentos recuperados: {len(docs)}:"]
        for d in details:
            s = (
                f"- {d['enfermedad']}: matched_cases={d['matched_cases']}/{d['total_docs']} "
                f"(db_ratio={d['db_ratio']}), symptom_support={d['symptom_support']}, score={d['score_pct']}%"
            )
            justification_lines.append(s)

        justification_lines.append(f"Métrica final (top candidato): {top_pct}%")
        justification = "\n".join(justification_lines)

        return {
            'metric': top_pct,
            'justification': justification,
            'supported_candidates': supported,
            'hallucinated_candidates': hallucinated
        }
