import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { decodeImageDataUrl, imageBytesToDataUrl } from "@/lib/server/image-utils";
import { getImageBytes } from "@/lib/server/image-store";

export interface VectorizeRequest {
  filename?: string;
  imageDataUrl?: string;
  includeMeta?: boolean;
  debugDump?: boolean;
}

export interface VectorizeResult {
  success: boolean;
  svg: string;
  svgFilename: string;
  traceMeta?: Record<string, unknown>;
  debugSvgFilename?: string;
}

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

function resolveImageBytes(options: VectorizeRequest): Buffer {
  if (options.imageDataUrl) {
    return decodeImageDataUrl(options.imageDataUrl);
  }
  if (options.filename) {
    const bytes = getImageBytes(options.filename);
    if (bytes) return bytes;
  }
  throw new Error("Either filename or image_data_url is required");
}

function runPythonVectorize(
  imageBytes: Buffer,
  options: Pick<VectorizeRequest, "includeMeta" | "debugDump">
): Promise<VectorizeResult> {
  const projectRoot = process.cwd();
  const scriptPath = path.join(projectRoot, "scripts", "vectorize_cli.py");
  const python = resolvePythonExecutable();

  const payload = JSON.stringify({
    image_base64: imageBytes.toString("base64"),
    include_meta: options.includeMeta ?? false,
    debug_dump: options.debugDump ?? false,
  });

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

    child.stdout.on("data", (chunk: Buffer) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr += chunk.toString();
    });

    child.on("error", (error) => {
      reject(
        new Error(
          `Failed to start Python (${python}): ${error.message}. ` +
            "Install Python dependencies with: pip install -r requirements.txt"
        )
      );
    });

    child.on("close", (code) => {
      if (!stdout.trim()) {
        reject(
          new Error(
            stderr.trim() ||
              `Vectorization failed (exit ${code ?? "unknown"}). ` +
                "Ensure Python deps are installed: pip install -r requirements.txt"
          )
        );
        return;
      }

      try {
        const parsed = JSON.parse(stdout) as {
          error?: string;
          success?: boolean;
          svg?: string;
          svg_filename?: string;
          trace_meta?: Record<string, unknown>;
          debug_svg_filename?: string;
        };

        if (parsed.error) {
          reject(new Error(parsed.error));
          return;
        }
        if (!parsed.svg) {
          reject(new Error("Vectorization returned no SVG data"));
          return;
        }

        resolve({
          success: true,
          svg: parsed.svg,
          svgFilename: parsed.svg_filename || `vector_${Date.now()}.svg`,
          traceMeta: parsed.trace_meta,
          debugSvgFilename: parsed.debug_svg_filename,
        });
      } catch {
        reject(
          new Error(
            `Invalid vectorization output: ${stdout.slice(0, 200) || stderr.slice(0, 200)}`
          )
        );
      }
    });

    child.stdin.write(payload);
    child.stdin.end();
  });
}

async function proxyToFlask(
  body: Record<string, unknown>
): Promise<VectorizeResult | null> {
  const baseUrl =
    process.env.FLASK_API_URL?.replace(/\/$/, "") || "http://127.0.0.1:5002";

  try {
    const response = await fetch(`${baseUrl}/api/vectorize-image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(120_000),
    });

    if (!response.ok) {
      return null;
    }

    const data = (await response.json()) as {
      error?: string;
      success?: boolean;
      svg?: string;
      svg_filename?: string;
      trace_meta?: Record<string, unknown>;
      debug_svg_filename?: string;
    };

    if (!data.svg) {
      return null;
    }

    return {
      success: true,
      svg: data.svg,
      svgFilename: data.svg_filename || `vector_${Date.now()}.svg`,
      traceMeta: data.trace_meta,
      debugSvgFilename: data.debug_svg_filename,
    };
  } catch {
    return null;
  }
}

export async function vectorizeImage(
  options: VectorizeRequest
): Promise<VectorizeResult> {
  const imageBytes = resolveImageBytes(options);

  const flaskBody: Record<string, unknown> = {
    include_meta: options.includeMeta ?? false,
    debug_dump: options.debugDump ?? false,
  };
  if (options.imageDataUrl) {
    flaskBody.image_data_url = options.imageDataUrl;
  } else if (options.filename) {
    flaskBody.image_data_url = imageBytesToDataUrl(imageBytes);
    flaskBody.filename = options.filename;
  }

  const flaskResult = await proxyToFlask(flaskBody);
  if (flaskResult) {
    return flaskResult;
  }

  return runPythonVectorize(imageBytes, options);
}
