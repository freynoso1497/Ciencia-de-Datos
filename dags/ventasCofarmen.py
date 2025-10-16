from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pathlib import Path
import os, subprocess

def in_container() -> bool:
    return os.path.exists("/.dockerenv") or os.path.isdir("/.dockerinit")

def get_output_dirs():
    """
    Donde dejamos el archivo en un lugar visible desde tu PC:
      - main_out: ./output (montado como /usr/local/airflow/output)
      - mirror_out: ./include (por si el Finder tarda en refrescar ./output)
    """
    if in_container():
        project_root = Path("/usr/local/airflow")
    else:
        # Ejecutando Airflow sin Docker → usá tu carpeta local directamente
        project_root = Path.home() / "Documents" / "proyecto"

    main_out = Path(os.environ.get("VENTAS_OUTPUT_DIR", project_root / "output"))
    mirror_out = project_root / "include"
    return main_out, mirror_out

default_args = {
    "owner": "Facundo y Lucila",
    "start_date": datetime.today() - timedelta(days=1),
    "retries": 0,
    "email_on_failure": False,
    "depends_on_past": False,
}

def export_query_to_csv(**kwargs):
    import psycopg2, csv, shutil

    ds = kwargs.get("ds", datetime.today().strftime("%Y-%m-%d"))
    main_out, mirror_out = get_output_dirs()
    main_out.mkdir(parents=True, exist_ok=True)
    mirror_out.mkdir(parents=True, exist_ok=True)

    csv_name = f"ventas_{ds}.csv"
    path_main = main_out / csv_name
    path_mirror = mirror_out / csv_name

    # Conexión Postgres (dentro de Astro: host.docker.internal; fuera: localhost)
    host = os.environ.get("VENTAS_DB_HOST", "host.docker.internal" if in_container() else "localhost")
    port = int(os.environ.get("VENTAS_DB_PORT", "5434"))
    db   = os.environ.get("VENTAS_DB_NAME", "mydb")
    user = os.environ.get("VENTAS_DB_USER", "admin")
    pwd  = os.environ.get("VENTAS_DB_PASS", "admin")

    query = """
        select 
            v."Documento"::bigint::text,
            v."CreadoEl",
            v."Hora",
            v."CreadoPor",
            v."ClaseVenta",
            c."Denominacion" as "DenominacionClase",
            v."ValorNeto",
            v."Moneda",
            v."GrupoVendedores",
            b."Denominacion" as "Vendedor",
            v."Oficina",
            v."PedidoCliente",
            v."Telefono",
            v."Cliente"::int::text,
            c2."Nombre1",
            C2."Nombre2",
            c2."Poblacion" as "Localidad",
            c2."CodPostal",
            c2."Calle",
            c2."CUIT"::bigint::text
            from Ventas v
            inner join vendedores b on b."Vendedores" = v."GrupoVendedores" 
            inner join clasesdocventas c on c."Clase" = v."ClaseVenta"
            inner join clientes c2 on  c2."Cliente" = v."Cliente"::text
    """  # :contentReference[oaicite:1]{index=1}

    conn = psycopg2.connect(host=host, port=port, database=db, user=user, password=pwd)
    cur = conn.cursor()
    # Volcamos el resultado a CSV sin cargar todo en memoria (2M+ filas OK)
    copy_sql = f"COPY ({query}) TO STDOUT WITH CSV HEADER"
    with open(path_main, "w") as f:
        cur.copy_expert(copy_sql, f)
    cur.close(); conn.close()


    # Copia espejo a ./include (para que lo veas sí o sí en el host)
    if path_main.resolve() != path_mirror.resolve():
        try:
            shutil.copy2(path_main, path_mirror)
            print(f"[INFO] Copia espejo: {path_mirror}")
        except Exception as e:
            print(f"[WARN] No se pudo copiar a include/: {e}")

    # Permisos amigables en host (por si tu FS es quisquilloso)
    try:
        os.chmod(path_main, 0o666)
        if path_mirror.exists():
            os.chmod(path_mirror, 0o666)
    except Exception as e:
        print(f"[WARN] No se pudieron ajustar permisos: {e}")

    # Listado para que lo veas en los logs
    try:
        print("[INFO] Contenido de ./output:")
        subprocess.run(["ls", "-lh", str(main_out)], check=False)
        print("[INFO] Contenido de ./include:")
        subprocess.run(["ls", "-lh", str(mirror_out)], check=False)
    except Exception:
        pass

    print(f"[INFO] CSV principal: {path_main}")
    return str(path_main)

    chunksize = 50000
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        for i, chunk in enumerate(pd.read_sql(query, engine, chunksize=chunksize)):
            # header solo en el primer chunk
            chunk.to_csv(f, index=False, header=(i == 0))
            print(f"Exportado chunk {i+1} con {len(chunk)} filas")

with DAG(
    dag_id="ventasCofarmen",
    description="Exporta ventas a CSV en ./output (visible en tu PC)",
    default_args=default_args,
    schedule=None,
    catchup=False,
    tags=["cofarmen", "ventas", "local"],
) as dag:

    export_task = PythonOperator(
        task_id="export_query_to_csv",
        python_callable=export_query_to_csv,
        do_xcom_push=True,
    )
