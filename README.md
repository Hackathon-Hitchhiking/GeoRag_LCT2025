# GeoRAG Localization Service

GeoRAG — это промышленный сервис визуальной геолокации, построенный поверх идей
[HLOC](https://github.com/cvg/Hierarchical-Localization). Он сочетает глобальное
ранжирование NetVLAD и локальное сопоставление SuperPoint/LightGlue, сохраняет
артефакты в S3, а косинусное ранжирование выполняет в Qdrant. Метаданные и
служебная информация остаются в PostgreSQL. Сервис умеет обогащать ответы
адресами через Nominatim и готов к масштабированию на сотни тысяч фотографий
без прогрева всех дескрипторов в оперативной памяти.

## Ключевые преимущества

- **Полный пайплайн VPR из HLOC.** Глобальные дескрипторы NetVLAD отбирают
  кандидатов, LightGlue и RANSAC (USAC MAGSAC) подтверждают соответствия на
  уровне ключевых точек и геометрии, формируя итоговую метрику уверенности.
- **Расширяемая архитектура.** Приложение разделено на `core`, `application` и
  `infrastructure`: модели и конфигурация отделены от сервисов и адаптеров.
- **Максимум сигналов.** Мы сохраняем и используем глобальные и локальные
  признаки, количество соответствий, нормированные оценки и статистики для
  последующего анализа качества.
- **S3 + Qdrant + PostgreSQL.** Исходники и признаки хранятся в объектном
  хранилище, глобальные векторы лежат в Qdrant, а реляционная база содержит
  только ссылки и метаданные.
- **3D-точки для фронтенда.** Геометрическая верификация восстанавливает
  относительную позу камеры и триангулирует облако точек, поэтому фронтенд
  получает готовую 3D-структуру объекта и матрицу перехода между кадрами.
- **Стриминг признаков.** Локальные дескрипторы подкачиваются из S3 по требованию
  и кешируются в LRU, что позволяет работать с десятками тысяч изображений без
  дополнительного прогрева индекса.
- **Проработанное API.** REST-эндпоинты позволяют загружать снимки, искать
  совпадения, получать 3D-точки и результаты обратного геокодирования.
- **Инструменты разработчика.** Makefile, CLI-скрипт для пакетной загрузки,
  интерактивный `scripts/plot_point_cloud.py` для инспекции реконструкций,
  готовые Dockerfile/compose и набор статических проверок (`ruff`, `mypy`).

## Архитектура проекта

```
app/
├── core/                # конфигурация и базовые сущности
├── application/         # прикладные сервисы (ingestion, search, matcher)
├── infrastructure/      # адаптеры к PostgreSQL и S3
├── api/                 # FastAPI-роуты и схемы
└── utils/               # вспомогательные утилиты
```

1. **Ингест**: изображение поступает через API или CLI, вычисляются локальные и
   глобальные признаки, пакеты сериализуются в S3, а в Qdrant создаётся точка с
   нормализованным вектором и ссылками. PostgreSQL хранит только служебные поля
   (ключи, координаты, метаданные).
2. **Поиск**: NetVLAD вычисляет глобальный дескриптор запроса, косинусная
   близость выбирает кандидатов из Qdrant, LightGlue подтверждает совпадения и
   формирует итоговый скоринг.
3. **Обратное геокодирование** (опционально) добавляет адрес по координатам.

## Быстрый старт

### 1. Подготовка окружения

```bash
cp .env.example .env
uv sync
```

> Для GPU-окружений установите PyTorch с нужной сборкой CUDA и задайте
> `COMPUTE_DEVICE=cuda` (или `cuda:0`) в переменных окружения.

### 2. Запуск сервиса

Запуск в режиме разработки:

```bash
uv run uvicorn app.main:app --reload
```

Запуск в Docker:

```bash
docker compose up --build
```

### 3. Массовая загрузка датасета

1. Скачайте уличные снимки Московской области из Mapillary (требуется токен):

   ```bash
   docker compose run --rm app make bootstrap-moscow
   ```

   Скрипт `scripts/download_moscow_mapillary.py` создаст каталог
   `train_data/mapillary_moscow` и заполнит его изображениями с метаданными в
   `metadata.jsonl`.

2. При необходимости добавьте собственные датасеты в `train_data/`, соблюдая
   формат, описанный ниже.

3. Запустите пакетный ingest:

   ```bash
   docker compose run --rm app make ingest
   ```

   `scripts/ingest_train_data.py` вычислит признаки и наполнит базу. Хостовый
   каталог `train_data/` монтируется в контейнер автоматически, поэтому пути и
   переменные окружения из `.env` совпадают с локальными. Одновременно ведётся
   потоковое логирование в `logs/ingest.log`, чтобы отслеживать прогресс без
   подключения к контейнеру.

### Формат каталога `train_data`

Сервис ожидает структуру, совместимую с несколькими источниками данных:

```
train_data/
├── mapillary_moscow/
│   ├── metadata.jsonl            # построчный JSON с относительными путями
│   ├── 1234567890.jpg            # само изображение (jpg/jpeg/png)
│   ├── 1234567890.json           # (опц.) sidecar c метаданными и GPS
│   └── ...
└── custom_dataset/
    ├── image_a.png
    ├── image_a.meta.json        # (опц.) широта/долгота/произвольные поля
    ├── manifest.jsonl           # (опц.) записи вида {"filename": "custom_dataset/image_a.png", ...}
    └── metadata.json            # (опц.) агрегированный JSON с полем results
```

- **Обязательные поля:** `filename` (относительный путь), сама фотография.
- **Высокоприоритетные поля:** `latitude`, `longitude` (в WGS84). Они могут
  находиться либо в `metadata.jsonl/manifest.jsonl`, либо в соседнем JSON.
- **Дополнительные поля:** `captured_at`, `compass_angle`, произвольные метки —
  все они будут сохранены в PostgreSQL как JSON-метаданные.
- Поддерживаются форматы изображений: `jpg`, `jpeg`, `png`, `bmp`, `tif`, `tiff`.
- Если в каталоге присутствует агрегированный `metadata.json`, ingest ищет рядом
  файлы `<id>.<ext>` и извлекает поля `speed`, `angle`, `create_timestamp`,
  `device`, `camera`. Поля `captured_at` и `compass_angle` автоматически
  нормализуются в `create_timestamp` и `angle`.

Скрипт ingest объединяет данные из манифеста и sidecar-файлов, приоритет у
метаданных, лежащих рядом с изображением.

## Переменные окружения

Полный перечень значений приведён в [.env.example](./.env.example). Важные
параметры:

| Переменная | Назначение |
|------------|------------|
| `DATABASE_DSN` | Строка подключения к PostgreSQL (`asyncpg`). |
| `QDRANT_*` | URL, API-ключ, коллекция и настройки шардинга/репликации Qdrant. |
| `S3_*`, `STORAGE_*` | Параметры S3-совместимого хранилища и префиксы каталогов. |
| `FEATURE_MAX_KEYPOINTS`, `FEATURE_CACHE_*`, `FEATURE_PREFETCH_LIMIT` | Ограничения SuperPoint и размеры кеша локальных признаков. |
| `GLOBAL_DESCRIPTOR_TYPE`, `LOCAL_FEATURE_TYPE`, `MATCHER_TYPE` | Используемые модели пайплайна. |
| `RETRIEVAL_CANDIDATES`, `GLOBAL_SCORE_WEIGHT`, `LOCAL_SCORE_WEIGHT`, `GEOMETRY_SCORE_WEIGHT` | Настройки ранжирования поиска и веса уровней пайплайна. |
| `POINT_CLOUD_LIMIT` | Максимальное количество нормализованных лучей в API-ответе. |
| `COMPUTE_DEVICE` | Явно задаёт устройство (`cpu`, `cuda`, `mps`, `cuda:0` и т.п.). |
| `NOMINATIM_USER_AGENT` | Пользовательский агент для прямого и обратного геокодирования. |
| `MAPILLARY_TOKEN` | Токен Mapillary для загрузки эталонных данных Московской области. |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_SECONDS` | Параметры визуально-языковой модели. |

## Визуально-языковой модуль Ollama

Сервис включает отдельный адаптер для визуально-языковой модели
Qwen2.5-VL-7B-Instruct, развёрнутой через [Ollama](https://ollama.com/).
Контейнер `ollama` добавлен в `docker-compose.yml` и по умолчанию слушает порт
`11434`.

Чтобы подготовить модель, выполните один раз:

```bash
docker compose up -d ollama
docker compose exec ollama ollama pull qwen2.5-vl:7b-instruct
```

Высокоуровневый интерфейс расположен в `app/application/vlm.py`. Он описывает
уличную сцену и формирует JSON с двумя полями: `address` (если в кадре есть
точный адрес или указатель, иначе `null`) и `description` с кратким описанием
местности.

Пример использования внутри приложения:

```python
from app.application.vlm import VisionLanguageAnalyzer
from app.infrastructure.vlm import OllamaVLMClient

client = OllamaVLMClient()
analyzer = VisionLanguageAnalyzer(client)
scene = await analyzer.describe(image_base64)
print(scene.address, scene.description)
```

## API

После запуска сервис доступен по `/docs`. Основные маршруты:

- `PUT /v1/images` — загрузить изображение и вычислить признаки.
- `POST /v1/search_by_image` — найти топ-N совпадений; параметр `plot_dots=true`
  добавляет нормализованные 3D-лучи и соответствия LightGlue в ответ.
- `POST /v1/search_by_coordinates` — вернуть ближайшие снимки к заданным
  координатам (ответ содержит расстояние в метрах).
- `POST /v1/search_by_address` — геокодировать адрес через Nominatim и вернуть
  ближайшие изображения по координатам адреса.
- `GET /v1/health` — проверка готовности сервиса.

Пример ответа поиска:

```json
{
  "query_image_url": "https://s3.local/georag/queries/7a8b....jpg",
  "query_point_cloud": [
    {"x": -1.72, "y": 0.64, "z": 6.81, "score": 0.92},
    {"x": -1.31, "y": -0.42, "z": 5.94, "score": 0.88}
  ],
  "matches": [
    {
      "image_id": 42,
      "confidence": 0.84,
      "global_similarity": 0.67,
      "local_matches": 542,
      "local_match_ratio": 0.46,
      "local_mean_score": 0.88,
      "geometry_inliers": 311,
      "geometry_inlier_ratio": 0.57,
      "geometry_score": 0.50,
      "image_url": "https://s3.local/georag/images/42a....jpg",
      "geometry_score": 0.50,
      "image_url": "https://s3.local/georag/images/42a....jpg",
      "latitude": 55.75,
      "longitude": 37.61,
      "address": "Москва, Россия",
      "metadata": {"source": "moscow_dataset"},
      "point_cloud": [
        {"x": -1.65, "y": 0.71, "z": 6.22, "score": 0.90},
        {"x": -1.17, "y": -0.31, "z": 5.46, "score": 0.85}
      ],
      "correspondences": [
        {
          "query": {"x": -1.72, "y": 0.64, "z": 6.81, "score": 0.92},
          "candidate": {"x": -1.65, "y": 0.71, "z": 6.22, "score": 0.90},
          "score": 0.93
        }
      ],
      "relative_rotation": [
        0.99, -0.02, 0.10,
        0.01,  1.00, 0.03,
       -0.10, -0.02, 0.99
      ],
      "relative_translation": [0.12, -0.04, 0.01]
    }
  ]
}
```

Поля `relative_rotation` и `relative_translation` описывают переход из системы
координат запроса в систему кандидата и позволяют восстановить сцену в единой
рамке. `point_cloud` содержит триангулированные точки в соответствующей системе
координат (для запроса — в его собственной, для кандидата — после применения
перехода).

Пример ответа на `POST /v1/search_by_coordinates`:

```json
{
  "matches": [
    {
      "image_id": 21,
      "distance_meters": 37.4,
      "image_url": "https://s3.local/georag/images/21a....jpg",
      "latitude": 55.752,
      "longitude": 37.617,
      "address": "Москва, ул. Тверская",
      "metadata": {"dataset": "moscow"},
      "point_cloud": [
        {"x": -0.09, "y": 0.03, "z": 0.99, "score": 0.78}
      ]
    }
  ]
}
```

## Визуализация реконструкций

Для отладки фронтенда и проверки качества 3D-точек воспользуйтесь скриптом
`scripts/plot_point_cloud.py`. Он принимает JSON-ответ поиска и визуализирует
облака точек запроса и выбранного совпадения в единой системе координат.

```bash
uv run python scripts/plot_point_cloud.py search_response.json --match-index 0 --limit 1500 --save pointcloud.png
```

Если аргумент `--save` не указан, скрипт откроет интерактивное окно Matplotlib.

## Проверки качества и форматирование

```bash
make lint       # ruff
make typecheck  # mypy
```

## Дополнительные заметки

- NetVLAD, SuperPoint и LightGlue автоматически загружают предобученные веса при
  первом запуске и кэшируют их в каталоге `~/.cache/torch/hub`.
- Геометрическая проверка строится через `cv2.findFundamentalMat` (USAC MAGSAC)
  — это повторяет этап локализации HLOC и повышает точность.
- Для интеграции с новыми моделями достаточно реализовать новые адаптеры в
  `app/application` и внедрить их через конфигурацию.
- При работе в продакшне рекомендуется вынести очередь обработки (например,
  Celery) и настроить отказоустойчивое S3-хранилище (MinIO, Yandex Object
  Storage и т.п.).
