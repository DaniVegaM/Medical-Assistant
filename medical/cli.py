import json
from .system import MedicalDiagnosticSystem
from .evaluator import DiagnosisEvaluator


def prompt_input(prompt_text, default=None):
    val = input(prompt_text + (" [Enter para omitir]" if default is not None else "") + ": ")
    if val.strip() == "" and default is not None:
        return default
    return val.strip()


def run_cli():
    """Interfaz mínima para que el médico introduzca síntomas y confirme el diagnóstico."""
    print("🏥 Sistema de diagnóstico - Entrada de síntomas")

    # 1) Pedir síntomas principales (lista)
    sintomas_raw = input("Introduce los síntomas principales (separados por coma):\n> ")
    sintomas = [s.strip() for s in sintomas_raw.split(",") if s.strip()]

    # 2) Preguntas básicas que el sistema solicita
    edad = prompt_input("Edad del paciente (años)", default="?")
    genero = prompt_input("Género (M/F/O)", default="?")

    tiene_antecedentes = prompt_input("¿El paciente tiene antecedentes médicos relevantes? (s/n)", default="n")
    antecedentes = ""
    if tiene_antecedentes.lower().startswith("s"):
        antecedentes = input("Describe los antecedentes (breve):\n> ")

    extras_raw = prompt_input("¿Deseas añadir síntomas extras generales? (separa por coma)", default="")
    sintomas_extras = [s.strip() for s in extras_raw.split(",") if s.strip()]

    # Construir caso
    caso = {
        "id": "CLI-CASE-1",
        "paciente": {
            "edad": edad,
            "genero": genero,
            "sintomas": sintomas + sintomas_extras,
            "duracion": "desconocida",
            "progresion": "desconocida"
        },
        "antecedentes": antecedentes
    }

    # Inicializar sistema
    sistema = MedicalDiagnosticSystem()

    # Diagnosticar
    resultado = sistema.diagnosticar(caso)

    # Evaluar el diagnóstico respecto a la DB
    evaluator = DiagnosisEvaluator()
    eval_res = evaluator.evaluate(resultado, caso, sistema)

    print("\n� Métrica y justificación del evaluador:")
    print(eval_res.get('justification', 'Sin justificación.'))

    # Decide si el top resultado está soportado por la DB
    diag = resultado.get("diagnostico_final", {})
    top_name = str(diag.get('enfermedad', '')).strip()

    # Si no hay DB para evaluar, mostrar resultado y preguntar al médico como antes
    if eval_res.get('metric') is None:
        print("\n⚠️ No es posible evaluar el resultado contra la base (DB ausente). Se muestra el diagnóstico igual).")
        print(json.dumps(diag, indent=2, ensure_ascii=False))
        confirm = prompt_input("¿Es correcto el diagnóstico? (s/n)", default="s")
        if confirm.lower().startswith("s"):
            print("✅ Diagnóstico confirmado. Gracias.")
            # Guardar el caso confirmado en la BD para mejorar la métrica futura
            diag_name = top_name or prompt_input("Confirma el nombre del diagnóstico para guardar:", default=top_name)
            sistema.save_case(caso, diag_name)
            # Generar 20 casos sintéticos para reforzar la BD
            nuevos = sistema.learn_from_feedback(caso, diag_name, n=20)
            print(f"Se generaron {len(nuevos)} casos sintéticos y se añadieron a la BD.")
        else:
            real = prompt_input("Introduce el diagnóstico real (nombre):")
            print("Generando y guardando casos sintéticos para aprender del error...")
            casos_generados = sistema.learn_from_feedback(caso, real, n=5)
            # Guardar también el caso original etiquetado con el diagnóstico real
            sistema.save_case(caso, real)
            print(f"Se generaron {len(casos_generados)} casos y se guardaron (si vectorstore disponible).")
    else:
        hallucinated = eval_res.get('hallucinated_candidates', [])
        # build list of hallucinated names
        halluc_names = [str(c.get('enfermedad','')).strip().lower() for c in hallucinated]

        if top_name.lower() in halluc_names:
            # Top prediction is not supported by DB -> hide it and ask the doctor
            print('\n⚠️ El diagnóstico propuesto NO está soportado por la base de conocimiento (posible alucinación).')
            print('El sistema declara que no sabe a qué enfermedad corresponde basado en la DB.')
            real = prompt_input("Por favor, introduce el diagnóstico correcto para que el sistema aprenda:")
            print("Generando y guardando casos sintéticos para aprender del diagnóstico correcto...")
            casos_generados = sistema.learn_from_feedback(caso, real, n=5)
            print(f"Se generaron {len(casos_generados)} casos y se guardaron (si vectorstore disponible).")
        else:
            # Top is supported -> mostrar y confirmar (pero sustituimos la probabilidad por la métrica del evaluador)
            print("\n🔍 Resultado propuesto por el sistema (soportado por DB):")
            # Reemplazar la probabilidad del LLM por la métrica del evaluador (porcentaje -> 0-1)
            metric_pct = eval_res.get('metric')
            if metric_pct is not None:
                try:
                    diag['probabilidad'] = round(metric_pct / 100.0, 3)
                except Exception:
                    pass
            print(json.dumps(diag, indent=2, ensure_ascii=False))
            confirm = prompt_input("¿Es correcto el diagnóstico? (s/n)", default="s")
            if confirm.lower().startswith("s"):
                print("✅ Diagnóstico confirmado. Gracias.")
                sistema.save_case(caso, top_name)
                nuevos = sistema.learn_from_feedback(caso, top_name, n=20)
                print(f"Se generaron {len(nuevos)} casos sintéticos y se añadieron a la BD.")
            else:
                real = prompt_input("Introduce el diagnóstico real (nombre):")
                print("Generando y guardando casos sintéticos para aprender del error...")
                casos_generados = sistema.learn_from_feedback(caso, real, n=5)
                sistema.save_case(caso, real)
                print(f"Se generaron {len(casos_generados)} casos y se guardaron (si vectorstore disponible).")

    print("Proceso finalizado.")


if __name__ == '__main__':
    run_cli()
