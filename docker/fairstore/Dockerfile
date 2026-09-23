# ckan-dev runs Flask's debug server (with the debug toolbar) for local work.
# A deployed Fair Store builds from ckan-base, the uwsgi production image;
# both share the same start scripts and /docker-entrypoint.d hooks.
ARG CKAN_BASE_IMAGE=ckan/ckan-dev:2.11.6
FROM ${CKAN_BASE_IMAGE}

# Keep the extension revision deterministic and aligned with the CKAN 2.11
# version used by the hosted Fair Store. This revision's CI covers CKAN 2.11.
ARG CKANEXT_HIERARCHY_SHA=53c1ee74805d79aef7c9c01cc513eb881cb8928f

USER root
RUN pip install --no-cache-dir \
    "git+https://github.com/ckan/ckanext-hierarchy.git@${CKANEXT_HIERARCHY_SHA}#egg=ckanext-hierarchy"
COPY pin-secrets.sh /docker-entrypoint.d/10-pin-secrets.sh
COPY templates.sh /docker-entrypoint.d/20-templates.sh
COPY secure-cookies.sh /docker-entrypoint.d/30-secure-cookies.sh
COPY templates /srv/app/fairstore_templates

USER ckan
