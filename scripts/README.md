# BBX Utility Scripts

Все скрипты для управления и тестирования BBX системы.

## 🚀 Запуск UI

### `start_ui.bat`
Запуск через Docker Compose (рекомендуется)
```bash
scripts\start_ui.bat
```

### `start_backend_local.bat`
Запуск backend локально без Docker
```bash
scripts\start_backend_local.bat
```

### `start_frontend_local.bat`
Запуск frontend локально без Docker
```bash
scripts\start_frontend_local.bat
```

---

## 🤖 Генерация с Llama

### `generate_with_llama.bat`
Генерация приложений с локальным Llama
```bash
scripts\generate_with_llama.bat
```

### `generate_crazy_app.py`
Python скрипт для генерации (вызывается из .bat)
```bash
python scripts\generate_crazy_app.py
```

---

## 🔧 Отладка

### `debug_backend.bat`
Проверка статуса и логов backend
```bash
scripts\debug_backend.bat
```

### `show_backend_logs.bat`
Показать полные логи backend
```bash
scripts\show_backend_logs.bat
```

### `restart_backend.bat`
Перезапустить backend контейнер
```bash
scripts\restart_backend.bat
```

### `rebuild_backend.bat`
Пересобрать и перезапустить backend
```bash
scripts\rebuild_backend.bat
```

### `verify_system.py`
Проверка системных зависимостей
```bash
python scripts\verify_system.py
```

---

## 📝 Быстрые команды

**Полный перезапуск с rebuild:**
```bash
scripts\rebuild_backend.bat
```

**Просто перезапуск:**
```bash
scripts\restart_backend.bat
```

**Посмотреть что происходит:**
```bash
scripts\debug_backend.bat
```

**Сгенерить приложение:**
```bash
scripts\generate_with_llama.bat
```
