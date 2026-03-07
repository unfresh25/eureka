# Eureka Frontend

Chat UI del agente legal Eureka.

## Setup

### Desarrollo local
```bash
# Opción 1 — VS Code Live Server (extensión)
# Abrir frontend/ con Live Server en puerto 5500

# Opción 2 — Python
cd frontend/
python -m http.server 5500
```

Luego abrir: http://localhost:5500

> El backend debe estar corriendo en http://localhost:8000

## Estructura
```
frontend/
├── index.html          # Punto de entrada
├── styles/
│   └── main.css        # Diseño dark mode
└── src/
    └── main.js         # Lógica del chat (API client, rendering)
```
