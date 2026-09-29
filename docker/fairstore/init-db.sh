#!/bin/bash
# Fair Store DB bootstrap (#133).
#
# ckan/ckan-postgres-dev is a test-fixture image with hardcoded ckan_test /
# datastore_test names that ignores configuration, so the Fair Store runs a
# plain Postgres and creates exactly the users and databases CKAN needs here.
# Runs once, on an empty data directory.
#
# Passwords come from CKAN_DB_PASSWORD / DATASTORE_DB_PASSWORD. The defaults
# are local-only; a deployed Fair Store sets real ones (deploy/fairstore).
# They are passed as psql variables and quoted by psql (:'var'), not spliced
# into the SQL by the shell.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" \
    -v ckan_pw="${CKAN_DB_PASSWORD:-ckan}" \
    -v datastore_pw="${DATASTORE_DB_PASSWORD:-datastore}" <<-SQL
    CREATE USER ckan WITH PASSWORD :'ckan_pw';
    CREATE DATABASE ckan OWNER ckan;

    CREATE USER ckan_datastore_write WITH PASSWORD :'datastore_pw';
    CREATE USER ckan_datastore_read WITH PASSWORD :'datastore_pw';
    CREATE DATABASE datastore OWNER ckan_datastore_write;
SQL

# Read-only datastore user: can connect and select, cannot write. CKAN's
# datastore set-permissions step tightens this further at first run.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname datastore <<-SQL
    GRANT CONNECT ON DATABASE datastore TO ckan_datastore_read;
    GRANT USAGE ON SCHEMA public TO ckan_datastore_read;
    ALTER DEFAULT PRIVILEGES FOR USER ckan_datastore_write IN SCHEMA public
        GRANT SELECT ON TABLES TO ckan_datastore_read;
SQL
