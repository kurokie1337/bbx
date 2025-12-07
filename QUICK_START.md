# BBX Quick Start Guide

## 🚀 Запуск UI (BBX Console)

### Вариант 1: Docker (Рекомендуется)
```bash
# Просто запусти этот файл:
start_ui.bat
```

Или вручную:
```bash
cd bbx-console
docker-compose up -d
```

После запуска:
- **UI**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Вариант 2: Без Docker (Локально)

**Backend:**
```bash
cd bbx-console\backend
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend (в новом терминале):**
```bash
cd bbx-console\frontend
npm install
npm run dev
```

---

## 🤖 Генерация приложений с Llama

### Быстрый старт:
```bash
# Запусти этот файл:
generate_with_llama.bat
```

Или вручную:
```bash
python generate_crazy_app.py
```

### Что произойдет:
1. Проверка установки `llama-cpp-python`
2. Загрузка модели Qwen (если нужно, ~250MB)
3. Генерация workflow из текстового описания
4. Сохранение в `examples/crazy_app.bbx`

### Если нужно установить зависимости:
```bash
pip install llama-cpp-python
```

---

## 📝 Ручная генерация через Python

```python
from blackbox.ai.generator import WorkflowGenerator

# Инициализация
gen = WorkflowGenerator()

# Генерация
workflow = gen.generate(
    "Create a workflow that prints ASCII art",
    output_file="my_workflow.bbx"
)

print(workflow)
```

---

## 🧪 Проверка системы

```bash
python verify_system.py
```

Проверяет:
- ✅ llama-cpp-python установлен
- ✅ Модели скачаны
- ✅ WorkflowGenerator работает

---

## 🎯 Использование UI для генерации

1. Запусти UI: `start_ui.bat`
2. Открой http://localhost:3000
3. Найди раздел "Chat" или "LLM"
4. Введи промпт для генерации
5. Llama сгенерирует workflow

---

## ❓ Проблемы?

### Docker не запускается:
- Убедись что Docker Desktop установлен и запущен
- Проверь: `docker --version`

### Llama не работает:
```bash
pip install llama-cpp-python
```

### Модель не скачивается:
Модель скачается автоматически при первом запуске генератора.
Или вручную через CLI (если есть):
```bash
bbx model download qwen-0.5b
```
