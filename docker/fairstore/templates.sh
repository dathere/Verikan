# Load the Fair Store's template overrides (docker/fairstore/templates), which
# render the qsv AI summary on dataset pages. Sourced from /docker-entrypoint.d.
ckan config-tool "$CKAN_INI" "extra_template_paths = /srv/app/fairstore_templates"
