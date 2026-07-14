import { createReadStream, existsSync, readFileSync } from "node:fs";
import { extname, join, normalize } from "node:path";
import { createServer } from "node:http";
import { OpenAICompatibleSkillProvider } from "./src/agentic-thinking-agent.js";

const root = process.cwd();
const port = Number.parseInt(process.env.PORT ?? "4173", 10);

loadEnvFile();
process.env.AGENT_SKILL_API_KEY ??= process.env.openai_key ?? process.env.OPENAI_API_KEY;
process.env.AGENT_SKILL_ENDPOINT ??= "https://api.openai.com/v1/chat/completions";

const contentTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".css", "text/css; charset=utf-8"],
  [".json", "application/json; charset=utf-8"]
]);

function resolvePath(url) {
  const pathname = new URL(url, `http://localhost:${port}`).pathname;
  const requested = pathname === "/" ? "/public/index.html" : pathname;
  const filePath = normalize(join(root, requested));

  if (!filePath.startsWith(root)) {
    return null;
  }

  return filePath;
}

createServer(async (request, response) => {
  if (request.method === "POST" && request.url === "/api/propose-skill") {
    await handleSkillProposal(request, response);
    return;
  }

  const filePath = resolvePath(request.url);

  if (!filePath || !existsSync(filePath)) {
    response.writeHead(404);
    response.end("Not found");
    return;
  }

  response.writeHead(200, {
    "Content-Type": contentTypes.get(extname(filePath)) ?? "text/plain; charset=utf-8"
  });
  createReadStream(filePath).pipe(response);
}).listen(port, () => {
  console.log(`Visualizer running at http://localhost:${port}`);
});

async function handleSkillProposal(request, response) {
  try {
    const context = await readJson(request);
    const provider = new OpenAICompatibleSkillProvider();
    const spec = await provider.proposeSkill(context);

    response.writeHead(200, {
      "Content-Type": "application/json; charset=utf-8"
    });
    response.end(JSON.stringify({ ok: true, spec }));
  } catch (error) {
    response.writeHead(503, {
      "Content-Type": "application/json; charset=utf-8"
    });
    response.end(
      JSON.stringify({
        ok: false,
        error: error instanceof Error ? error.message : "Skill proposal failed."
      })
    );
  }
}

function readJson(request) {
  return new Promise((resolve, reject) => {
    let body = "";

    request.setEncoding("utf8");
    request.on("data", (chunk) => {
      body += chunk;
      if (body.length > 128_000) {
        reject(new Error("Request body too large."));
        request.destroy();
      }
    });
    request.on("end", () => {
      try {
        resolve(JSON.parse(body || "{}"));
      } catch {
        reject(new Error("Request body must be JSON."));
      }
    });
    request.on("error", reject);
  });
}

function loadEnvFile() {
  const envPath = join(root, ".env");
  if (!existsSync(envPath)) return;

  for (const line of readFileSync(envPath, "utf8").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;

    const separator = trimmed.indexOf("=");
    if (separator === -1) continue;

    const key = trimmed.slice(0, separator).trim();
    const rawValue = trimmed.slice(separator + 1).trim();
    if (!key || process.env[key] !== undefined) continue;

    process.env[key] = stripEnvQuotes(rawValue);
  }
}

function stripEnvQuotes(value) {
  if (
    (value.startsWith('"') && value.endsWith('"')) ||
    (value.startsWith("'") && value.endsWith("'"))
  ) {
    return value.slice(1, -1);
  }

  return value;
}
