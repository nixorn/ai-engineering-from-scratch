# Капстоун 16 — Автономный агент GitHub Issue-to-PR

> AWS Remote SWE Agents, Cursor Background Agents, облачный OpenAI Codex и Google Jules в 2026 году сходятся к одной форме продукта: пометь issue, получи PR. Агент запускается в облачной песочнице, проверяет прохождение тестов и публикует готовый к ревью PR с обоснованием. Самые сложные части: автоматически воспроизвести окружение сборки репозитория, не допустить утечки учётных данных, ввести бюджеты на уровне репозитория и гарантировать, что агент не может сделать force-push. В этом капстоуне вы создадите self-hosted версию и сравните её по стоимости и pass rate с hosted-альтернативами.

**Тип:** Капстоун  
**Языки:** Python (агент), TypeScript (GitHub App), YAML (Actions)  
**Пререквизиты:** Фаза 11 (LLM engineering), Фаза 13 (tools), Фаза 14 (agents), Фаза 15 (autonomous), Фаза 17 (infrastructure)  
**Задействованные фазы:** P11 · P13 · P14 · P15 · P17  
**Время:** 30 часов

## Проблема

Асинхронный облачный агент для кодинга — это отдельная категория продуктов по сравнению с интерактивными кодовыми агентами (капстоун 01). UX строится вокруг метки в GitHub. Вы помечаете issue как `@agent fix this`, в облачной песочнице поднимается воркер, клонирует репозиторий, запускает тесты, редактирует файлы, проверяет изменения и открывает PR с обоснованием агента в описании. Без интерактивного цикла, без терминала. AWS Remote SWE Agents, Cursor Background Agents, OpenAI Codex cloud, Google Jules и Factory Droids сходятся к этой модели.

Инженерные сложности вполне конкретны: воспроизведение окружения (агент должен собрать репозиторий с нуля без заранее закешированного dev-образа), нестабильные тесты (их нужно перезапускать или изолировать), ограничение учётных данных (GitHub App с минимальными fine-grained permissions), соблюдение бюджета на репозиторий в день и политика запрета force-push. Капстоун измеряет pass rate, стоимость и безопасность по сравнению с hosted-альтернативами.

## Концепция

Триггер — GitHub webhook (метка в issue или комментарий в PR). Диспетчер ставит задачу в очередь ECS Fargate или Lambda. Воркер подтягивает репозиторий в песочницу Daytona или E2B с универсальным Dockerfile, выведенным из репозитория (язык, фреймворк). Агент запускает цикл mini-swe-agent или SWE-agent v2 на Claude Opus 4.7 или GPT-5.4-Codex. Итерация: чтение кода, предложение фикса, применение патча, запуск тестов.

Проверка — это гейт. До открытия PR в песочнице должен пройти полный CI. Считается дельта покрытия; если она отрицательная ниже порога, PR всё равно открывается, но получает метку `needs-review`. Агент публикует обоснование в описании PR и добавляет тред `@agent`, где ревьюер может задавать уточняющие вопросы.

Безопасность задаётся двумя разными поверхностями GitHub: App выдаёт краткоживущий installation token с `workflows: read` и узкими правами на содержимое репозитория/PR; branch protection (а не права App) принудительно включает «никаких прямых записей в `main`» и «никакого force-push» — App никогда не добавляется в bypass list. Path-scoped read-only доступ к `.github/workflows` не является реальным примитивом GitHub App, поэтому allow-list на редактирование файлов должен проверяться на уровне воркера. Ограничения бюджета на репозиторий в день применяются диспетчером (например, максимум 5 PR на репозиторий в день, $20 за PR).

## Архитектура

```
GitHub issue с меткой `@agent fix` или комментарий в PR
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (или GitHub Actions self-hosted runner)
       - получить репозиторий
       - вывести Dockerfile (язык, пакетный менеджер)
       - Daytona / E2B песочница с нужным runtime
       - clone -> git worktree -> ветка агента
            |
            v
    цикл mini-swe-agent / SWE-agent v2
       Claude Opus 4.7 или GPT-5.4-Codex
       инструменты: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    проверка: CI проходит в песочнице + проверка дельты покрытия
            |
            v (verified)
    git push + открытие PR через GitHub App
       тело PR = обоснование + сводка diff + URL трассы
       метка: needs-review
            |
            v
    оператор делает ревью; может упомянуть агента для follow-up
```

## Стек

- Триггер: GitHub App с fine-grained token; webhook receiver через Lambda или Fly.io
- Воркер: ECS Fargate task (или GitHub Actions self-hosted runner)
- Песочница: Daytona devcontainer или E2B sandbox на задачу
- Цикл агента: baseline mini-swe-agent или SWE-agent v2 на Claude Opus 4.7 / GPT-5.4-Codex
- Извлечение: tree-sitter repo-map + ripgrep
- Верификация: полный CI в песочнице + gate по дельте покрытия
- Наблюдаемость: Langfuse с архивом трассы на каждый PR, ссылка в теле PR
- Бюджет: дневной лимит долларов на репозиторий; максимум PR на репозиторий в день

## Реализуйте

1. **GitHub App.** Fine-grained installation token: issues read+write, pull_requests write, contents read+write, workflows read. Branch protection (единственный механизм, который это обеспечивает) принудительно включает «no direct push to `main`» и «no force-push»; App не добавляется в bypass list. Воркер принудительно проверяет «no writes under `.github/workflows`» через allow-list на предлагаемый diff, так как права GitHub App не path-scoped.

2. **Webhook receiver.** Lambda-функция принимает webhooks по метке issue / комментарию PR. Фильтрует по метке `@agent fix this`. Кладёт задачи в SQS.

3. **Dispatcher.** Берёт задачи из SQS. Применяет бюджетные ограничения на репозиторий в день. Поднимает ECS Fargate task с URL репозитория, телом issue и новой Daytona-песочницей.

4. **Inference окружения.** Определите язык (Python, Node, Go, Rust) и пакетный менеджер (uv, pnpm, go mod, cargo). Генерируйте Dockerfile на лету, если его нет.

5. **Цикл агента.** mini-swe-agent или SWE-agent v2 с Claude Opus 4.7. Инструменты: ripgrep, tree-sitter repo-map, read_file, edit_file, run_tests, git. Жёсткие лимиты: $20 стоимости, 30 минут wall-clock, 30 ходов агента.

6. **Верификация.** После завершения цикла запустите в песочнице полный набор тестов. Посчитайте дельту покрытия через jacoco / coverage.py. Если CI красный: остановка, PR не открывать. Если покрытие падает более чем на 2%: открыть PR с меткой `needs-review`.

7. **Публикация PR.** Запушьте ветку агента. Откройте PR через GitHub API с: заголовком, обоснованием, сводкой diff, URL трассы, стоимостью, числом ходов.

8. **Гигиена учётных данных.** Воркер работает с краткоживущим installation token GitHub App. Логи очищаются от секретов до архивации.

9. **Оценка.** 30 внутренних issue с заранее заданными задачами разной сложности. Измерьте pass rate, качество PR (размер diff, стиль, покрытие), стоимость, задержку. Сравните с Cursor Background Agents и AWS Remote SWE Agents на тех же issue.

## Используйте

```
# на github.com
  - пользователь ставит метку `@agent fix this` на issue #842
  - через 14 минут появляется PR #1903
  - тело:
    > Исправлен NPE в widget.dedupe(), вызванный null-записью компаратора.
    > Добавлен regression test widget_test.go::TestDedupeNullComparator.
    > Дельта покрытия: +0.12%
    > Ходы: 7  Стоимость: $1.80  Трасса: langfuse:...
    > Метка: needs-review
```

## Выпустите

`outputs/skill-issue-to-pr.md` — итоговый артефакт. GitHub App + асинхронный облачный воркер, который превращает помеченные issue в готовые к ревью PR с ограниченной стоимостью и scoped credentials.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Pass rate на 30 issue | Сквозной успех (CI зелёный + покрытие в норме) |
| 20 | Качество PR | Размер diff, дельта покрытия, соответствие стилю |
| 20 | Стоимость и задержка на закрытую задачу | $ и wall-clock на PR |
| 20 | Безопасность | Scoped token, бюджет на репозиторий, no force-push, гигиена учётных данных |
| 15 | UX оператора | Комментарии-обоснования, возможность retry, follow-up через @-mention |
| **100** | | |

## Упражнения

1. Добавьте режим «исправить нестабильный тест»: метка `@agent stabilize-flake TestX` запускает тест 50 раз в песочнице и предлагает минимальное изменение для стабилизации.

2. Сравните стоимость с Cursor Background Agents на трёх общих issue. Отчитайте, какие инструменты и в чём выигрывают.

3. Реализуйте бюджетный дашборд: стоимость на репозиторий в день, стоимость на пользователя. Настройте алерты на аномалии.

4. Сделайте режим «dry-run», который открывает draft PR без запуска CI, чтобы ревьюеры могли дёшево посмотреть план.

5. Добавьте политику хранения: ветки PR старше 7 дней без мерджа автоматически удаляются.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|---------------------|--------------------------------|
| GitHub App | «Scoped bot identity» | Приложение с fine-grained permissions + краткоживущий installation token |
| Async cloud agent | «Background agent» | Неинтерактивный воркер, работающий в облачной песочнице, а не в терминале |
| Environment inference | «Синтез Dockerfile» | Определение языка + пакетного менеджера, генерация Dockerfile при отсутствии |
| Verification | «CI-in-sandbox» | Запуск полного набора тестов внутри воркера до открытия PR |
| Coverage delta | «Сохранение покрытия» | Изменение процента покрытия тестами от base-ветки к ветке агента |
| Per-repo budget | «Дневной потолок» | Лимит в долларах и количестве PR, применяемый диспетчером |
| Rationale | «Объяснение в теле PR» | Сводка агента о том, что и зачем изменено; обязательна в теле PR |

## Дополнительное чтение

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — канонический референс асинхронного облачного агента
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI-референс
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — коммерческая альтернатива
- [OpenAI Codex (cloud)](https://openai.com/codex) — hosted-конкурент
- [Google Jules](https://jules.google) — hosted-версия от Google
- [Factory Droids](https://www.factory.ai) — альтернативный коммерческий референс
- [Документация GitHub App](https://docs.github.com/en/apps) — scoped bot identity
- [Облачные песочницы Daytona](https://daytona.io) — референс песочницы
