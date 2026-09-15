# OpenCode project guidance

- Treat `C:\iclr-oc` as the project root. Do not use the parent drive root as project context.
- Keep project changes inside this repository unless the user asks otherwise.
- Inspect the existing files and documentation before choosing a language, framework, or build tools. This scaffold intentionally does not assume a technology stack.
- Keep credentials out of tracked files. Local `.env` files are ignored; never print secret values.
- Run checks documented by the project. If no checks exist yet, say so rather than inventing a test command.
