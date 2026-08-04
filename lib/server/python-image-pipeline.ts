import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";

function resolvePythonExecutable(): string {
  if (process.env.PYTHON_PATH) {
    return process.env.PYTHON_PATH;
  }
  if (process.env.CONDA_PREFIX) {
    return path.join(process.env.CONDA_PREFIX, "bin", "python");
  }
  const candidates = [
    process.env.HOME
      ? path.join(process.env.HOME, "miniconda3/envs/img/bin/python")
      : null,
    "/opt/miniconda3/envs/img/bin/python",
  ].filter((value): value is string => Boolean(value));
  for (const candidate of candidates) {
    if (existsSync(candidate)) return candidate;
  }
  return "python3";
}

export async function runPythonImageScript<T extends Record<string, unknown>>(
  scriptName: string,
  payload: Record<string, unknown>,
  options?: { timeoutMs?: number }
): Promise<T> {
  const projectRoot = process.cwd();
  const scriptPath = path.join(projectRoot, "scripts", scriptName);
  const python = resolvePythonExecutable();
  const timeoutMs = options?.timeoutMs ?? 300_000;

  return new Promise((resolve, reject) => {
    const child = spawn(python, [scriptPath], {
      cwd: projectRoot,
      env: {
        ...process.env,
        PYTHONPATH: projectRoot,
        PYTHONNOUSERSITE: "1",
      },
      stdio: ["pipe", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(
        new Error(
          `${scriptName} timed out after ${Math.round(timeoutMs / 1000)}s. ` +
            "Try again or use a smaller image."
        )
      );
    }, timeoutMs);

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      clearTimeout(timer);
      reject(
        new Error(
          `Failed to start Python (${python}): ${error.message}. ` +
            "Install dependencies: pip install -r requirements.txt"
        )
      );
    });

    child.on("close", (code) => {
      clearTimeout(timer);

      if (!stdout.trim()) {
        reject(
          new Error(
            stderr.trim() ||
              `${scriptName} failed (exit ${code ?? "unknown"}). ` +
                "Ensure Python deps are installed: pip install -r requirements.txt"
          )
        );
        return;
      }

      try {
        const parsed = JSON.parse(stdout) as T & { error?: string };
        if (parsed.error) {
          reject(new Error(parsed.error));
          return;
        }
        resolve(parsed);
      } catch {
        reject(
          new Error(
            `Invalid ${scriptName} output: ${stdout.slice(0, 300) || stderr.slice(0, 300)}`
          )
        );
      }
    });

    child.stdin.write(JSON.stringify(payload));
    child.stdin.end();
  });
}
