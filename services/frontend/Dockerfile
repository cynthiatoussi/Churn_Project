# ─────────────────────────────────────────────────────────────
# services/frontend/Dockerfile
# ─────────────────────────────────────────────────────────────
# Image ultra-légère nginx:alpine (~25Mo).
# Sert les fichiers statiques ET proxifie les appels API
# vers les services Kubernetes internes (pas de CORS).
# Build depuis la RACINE du projet (context = .)
# ─────────────────────────────────────────────────────────────
FROM nginx:alpine

# Copie la config nginx personnalisée
COPY services/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Copie le fichier HTML du frontend
COPY services/frontend/index.html /usr/share/nginx/html/index.html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD wget -q --spider http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
