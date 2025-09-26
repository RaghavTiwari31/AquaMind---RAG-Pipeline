import os
import netCDF4
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from netCDF4 import num2date
from datetime import datetime

# PostgreSQL connection parameters
conn_params = {
    'dbname': 'geo_data',
    'user': 'postgres',
    'password': 'admin',
    'host': '127.0.0.1',
    'port': 5432,
}


def netcdf_to_dataframe(nc_path):
    ds = netCDF4.Dataset(nc_path, 'r')
    print(f"Opened NetCDF file: {nc_path}")

    lat = ds.variables['LATITUDE'][:]
    lon = ds.variables['LONGITUDE'][:]
    pres = ds.variables['PRES'][:]
    temp = ds.variables['TEMP'][:]
    salinity = ds.variables['PSAL'][:]
    time_var = ds.variables['JULD']
    time_units = getattr(time_var, 'units', 'days since 1950-01-01')
    dates = num2date(time_var[:], units=time_units)

    records = []
    n_profiles = lat.shape[0]
    n_levels = pres.shape[1]

    for i in range(n_profiles):
        date_i = dates[i]
        # Convert cftime object to datetime.datetime
        if not isinstance(date_i, datetime):
            if hasattr(date_i, 'datetime'):
                date_i = date_i.datetime
            else:
                date_i = datetime(date_i.year, date_i.month, date_i.day,
                                  date_i.hour, date_i.minute, date_i.second)
        for j in range(n_levels):
            records.append((
                date_i,
                float(lat[i]),
                float(lon[i]),
                float(pres[i, j]),
                float(temp[i, j]),
                float(salinity[i, j])
            ))

    df = pd.DataFrame(records, columns=['time', 'latitude', 'longitude', 'pressure', 'temperature', 'salinity'])
    ds.close()
    df = df.dropna(subset=['latitude', 'longitude', 'pressure', 'temperature', 'salinity'])

    print(f"Converted {nc_path} → DataFrame with {len(df)} valid rows")
    return df


def insert_to_postgres(df, conn_params):
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()

    records = list(df.itertuples(index=False, name=None))
    sql = """
        INSERT INTO latest_data(time, latitude, longitude, pressure, temperature, salinity)
        VALUES %s
    """
    execute_values(cur, sql, records)

    conn.commit()
    cur.close()
    conn.close()
    print("Inserted", len(records), "rows into PostgreSQL.")


def process_directory(nc_dir, batch_size=10):
    nc_files = [os.path.join(nc_dir, f) for f in os.listdir(nc_dir) if f.endswith(".nc")]
    print(f"Found {len(nc_files)} NetCDF files in {nc_dir}")

    # Process in batches
    for i in range(0, len(nc_files), batch_size):
        batch_files = nc_files[i:i+batch_size]
        print(f"\n🔹 Processing batch {i//batch_size + 1}: {len(batch_files)} files")

        batch_df = pd.DataFrame()
        for nc_file in batch_files:
            try:
                df = netcdf_to_dataframe(nc_file)
                batch_df = pd.concat([batch_df, df], ignore_index=True)
            except Exception as e:
                print(f"⚠️ Failed to process {nc_file}: {e}")

        if not batch_df.empty:
            insert_to_postgres(batch_df, conn_params)
        else:
            print("⚠️ Skipping empty batch")


if __name__ == "__main__":
    nc_directory = r"D:\codes\sih\geo_data"
    process_directory(nc_directory, batch_size=10)
    print("\n✅ Ingestion workflow completed successfully.")
