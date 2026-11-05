"use strict";

const {
  app,
  BrowserWindow,
  shell,
  Menu,
  dialog,
  ipcMain,
  session,
  safeStorage,
  systemPreferences,
} = require("electron");

const path = require("path");
const { spawn, execFileSync } = require("child_process");
const http = require("http");
const https = require("https");
const fs = require("fs");
const crypto = require("crypto");

app.setName("Sxarf AI");
app.setPath("userData", path.join(app.getPath("appData"), "Sxarf AI"));

// ============================================================
// Application state and paths
// ============================================================

let mainWindow = null;
let pythonProcess = null;
let quitting = false;
let googleAuthData = null;
let googleSignInInProgress = false;
let backendExit = null;
let backendSpawnError = null;
let startupPromise = null;
let voiceProcess = null;
let bootstrapProcess = null;
let setupProgressText = "Starting Sxarf AI...";

const SERVER_PORT = 8741;
const SERVER_URL = `http://127.0.0.1:${SERVER_PORT}`;
const IS_PACKAGED = app.isPackaged;
const IS_WINDOWS = process.platform === "win32";

const APP_RESOURCES = IS_PACKAGED
  ? process.resourcesPath
  : __dirname;

// The runtime must live outside the packaged application.
// Packaged .app bundles are not a writable location.
const SUPPORT_DIR = app.getPath("userData");
const VENV_DIR = path.join(SUPPORT_DIR, "venv");
const VENV_PYTHON = IS_WINDOWS
  ? path.join(VENV_DIR, "Scripts", "python.exe")
  : path.join(VENV_DIR, "bin", "python");
const HF_HOME = path.join(SUPPORT_DIR, "huggingface");
const REQUIREMENTS_FILE = path.join(
  APP_RESOURCES,
  IS_WINDOWS ? "requirements-windows.txt" : "requirements.txt"
);

const SERVER_SCRIPT = path.join(
  APP_RESOURCES,
  "server",
  IS_WINDOWS ? "server_windows.py" : "server.py"
);

const SETUP_MARKER = path.join(
  SUPPORT_DIR,
  ".first-launch-setup-complete"
);

const STATE_FILE = path.join(
  app.getPath("userData"),
  "app-state.json"
);

const GOOGLE_AUTH_FILE = path.join(
  app.getPath("userData"),
  "google-auth.json"
);

const LOCAL_PROFILE_FILE = path.join(
  app.getPath("userData"),
  "local-account.json"
);

const GOOGLE_CLIENT_ID = (
  process.env.SXARF_GOOGLE_CLIENT_ID || ""
).trim();

const ALLOWED_MODELS = [
  "smollm2",
  "smollm3",
  "llama32",
  "lfm25",
  "gemma2",
  "phi35",
  "stablecode",
];

// Old saved state may still say "qwen3".
function migrateModelId(id) {
  return id === "qwen3" ? "llama32" : id;
}

const ALLOWED_PLANS = [
  "Free",
  "Pro",
  "Max",
  "Unlimited",
];

// ============================================================
// General helpers
// ============================================================

function atomicWriteJson(filePath, value) {
  fs.mkdirSync(path.dirname(filePath), {
    recursive: true,
  });

  const temporaryPath =
    `${filePath}.${process.pid}.` +
    `${crypto.randomBytes(6).toString("hex")}.tmp`;

  try {
    fs.writeFileSync(
      temporaryPath,
      JSON.stringify(value, null, 2),
      {
        encoding: "utf8",
        mode: 0o600,
        flag: "wx",
      }
    );

    fs.renameSync(temporaryPath, filePath);
  } catch (error) {
    try {
      if (fs.existsSync(temporaryPath)) {
        fs.unlinkSync(temporaryPath);
      }
    } catch (_) {}

    throw error;
  }
}

function isPlainObject(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}

function defaultAppState() {
  return {
    chats: [],
    memory: "",
    preferredModel: "llama32",
    settings: {
      agentsMode: "auto",
      maxTokens: 4096,
      temperature: 0.7,
      autoScroll: true,
      enterToSend: true,
      reduceMotion: false,
      theme: "dark",
    },
  };
}

// ============================================================
// App state persistence
// ============================================================

function sanitizeSettings(settings) {
  const source = isPlainObject(settings) ? settings : {};
  const temperature = Number(source.temperature);

  return {
    agentsMode: ["auto", "on", "off"].includes(source.agentsMode)
      ? source.agentsMode
      : "auto",

    maxTokens: [1024, 2048, 4096, 8192].includes(
      Number(source.maxTokens)
    )
      ? Number(source.maxTokens)
      : 4096,

    temperature: Number.isFinite(temperature)
      ? Math.min(1.2, Math.max(0, temperature))
      : 0.7,

    autoScroll: source.autoScroll !== false,
    enterToSend: source.enterToSend !== false,
    reduceMotion: Boolean(source.reduceMotion),

    theme: ["system", "dark", "light"].includes(source.theme)
      ? source.theme
      : "dark",
  };
}

function sanitizeChats(chats) {
  if (!Array.isArray(chats)) return [];

  return chats
    .slice(0, 100)
    .filter(chat =>
      isPlainObject(chat) &&
      typeof chat.id === "string" &&
      Array.isArray(chat.messages)
    )
    .map(chat => ({
      id: chat.id.slice(0, 160),

      title: typeof chat.title === "string"
        ? chat.title.slice(0, 160)
        : "New chat",

      updated: Number.isFinite(Number(chat.updated))
        ? Number(chat.updated)
        : 0,

      messages: chat.messages
        .slice(-2000)
        .filter(message =>
          isPlainObject(message) &&
          ["user", "assistant"].includes(message.role) &&
          typeof message.content === "string"
        )
        .map(message => ({
          role: message.role,
          content: message.content.slice(0, 500000),
        })),
    }));
}

function readAppState() {
  try {
    if (!fs.existsSync(STATE_FILE)) {
      return defaultAppState();
    }

    const parsed = JSON.parse(
      fs.readFileSync(STATE_FILE, "utf8")
    );

    return {
      chats: sanitizeChats(parsed.chats),

      memory: typeof parsed.memory === "string"
        ? parsed.memory.slice(0, 50000)
        : "",

      preferredModel: ALLOWED_MODELS.includes(
        migrateModelId(parsed.preferredModel)
      )
        ? migrateModelId(parsed.preferredModel)
        : "llama32",

      settings: sanitizeSettings(parsed.settings),
    };
  } catch (error) {
    console.error("Could not read app state:", error);
    return defaultAppState();
  }
}

function writeAppState(state) {
  if (!isPlainObject(state)) {
    throw new Error("Invalid app state.");
  }

  const safeState = {
    chats: sanitizeChats(state.chats),

    memory: typeof state.memory === "string"
      ? state.memory.slice(0, 50000)
      : "",

    preferredModel: ALLOWED_MODELS.includes(
      migrateModelId(state.preferredModel)
    )
      ? migrateModelId(state.preferredModel)
      : "llama32",

    settings: sanitizeSettings(state.settings),
  };

  atomicWriteJson(STATE_FILE, safeState);
  return true;
}

ipcMain.handle("app-state:load", () => readAppState());

ipcMain.handle("app-state:save", (_event, state) => {
  return writeAppState(state);
});

function stopVoiceProcess() {
  if (!voiceProcess) return;
  const child = voiceProcess;
  voiceProcess = null;
  if (child.exitCode === null && child.signalCode === null) child.kill();
}

ipcMain.handle("voice:speak", (_event, text) => {
  if (typeof text !== "string" || !text.trim() || text.length > 100_000) {
    throw new TypeError("Speech text must be a non-empty string no longer than 100,000 characters.");
  }
  if (process.platform !== "darwin") {
    throw new Error("Native offline speech is only available on macOS.");
  }

  stopVoiceProcess();
  return new Promise((resolve, reject) => {
    const child = spawn("/usr/bin/say", ["-r", "190", text], {
      stdio: "ignore",
    });
    voiceProcess = child;
    child.once("error", error => {
      if (voiceProcess === child) voiceProcess = null;
      reject(new Error(`Could not start macOS speech: ${error.message}`));
    });
    child.once("close", (code, signal) => {
      if (voiceProcess === child) voiceProcess = null;
      if (code === 0 || signal === "SIGTERM") resolve(true);
      else reject(new Error(`macOS speech exited with code ${code ?? "unknown"}.`));
    });
  });
});

ipcMain.handle("voice:stop-speaking", () => {
  stopVoiceProcess();
  return true;
});

// ============================================================
// HTTPS JSON requests
// ============================================================

function httpsJsonRequest(urlString, options = {}) {
  return new Promise((resolve, reject) => {
    let url;

    try {
      url = new URL(urlString);
    } catch (_) {
      reject(new Error("Invalid request URL."));
      return;
    }

    if (url.protocol !== "https:") {
      reject(new Error("OAuth requests must use HTTPS."));
      return;
    }

    const body = options.body || "";
    const headers = { ...(options.headers || {}) };

    if (body && !headers["Content-Length"]) {
      headers["Content-Length"] = Buffer.byteLength(body);
    }

    const request = https.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port || 443,
        path: `${url.pathname}${url.search}`,
        method: options.method || "GET",
        headers,
      },
      response => {
        let responseBody = "";

        response.setEncoding("utf8");

        response.on("data", chunk => {
          responseBody += chunk;
        });

        response.on("end", () => {
          let payload = {};

          try {
            payload = responseBody
              ? JSON.parse(responseBody)
              : {};
          } catch (_) {
            reject(
              new Error("Google returned an invalid response.")
            );
            return;
          }

          if (
            response.statusCode < 200 ||
            response.statusCode >= 300
          ) {
            const message =
              payload.error_description ||
              payload.error ||
              `Google request failed (${response.statusCode}).`;

            reject(new Error(String(message)));
            return;
          }

          resolve(payload);
        });
      }
    );

    request.setTimeout(20000, () => {
      request.destroy(
        new Error("Google request timed out.")
      );
    });

    request.on("error", reject);

    if (body) request.write(body);
    request.end();
  });
}

// ============================================================
// Local account and plan
// ============================================================

function readLocalAccountData() {
  try {
    const data = JSON.parse(
      fs.readFileSync(LOCAL_PROFILE_FILE, "utf8")
    );

    return isPlainObject(data) ? data : {};
  } catch (_) {
    return {};
  }
}

function readLocalProfile() {
  const data = readLocalAccountData();

  return {
    name: typeof data.name === "string"
      ? data.name
      : "Local Guest",

    email: "",
    picture: "",
    provider: "local",
  };
}

function readLocalPlan() {
  const data = readLocalAccountData();

  return ALLOWED_PLANS.includes(data.plan)
    ? data.plan
    : "Free";
}

// ============================================================
// Google OAuth token storage
// ============================================================

function loadGoogleAuthData() {
  if (googleAuthData) return googleAuthData;

  if (!safeStorage.isEncryptionAvailable()) {
    return null;
  }

  try {
    if (!fs.existsSync(GOOGLE_AUTH_FILE)) {
      return null;
    }

    const stored = JSON.parse(
      fs.readFileSync(GOOGLE_AUTH_FILE, "utf8")
    );

    if (typeof stored.encrypted !== "string") {
      return null;
    }

    const decrypted = safeStorage.decryptString(
      Buffer.from(stored.encrypted, "base64")
    );

    const parsed = JSON.parse(decrypted);

    if (!isPlainObject(parsed)) return null;

    googleAuthData = parsed;
    return googleAuthData;
  } catch (error) {
    console.error(
      "Could not decrypt saved Google session:",
      error
    );

    return null;
  }
}

function saveGoogleAuthData(data) {
  googleAuthData = data;

  if (!safeStorage.isEncryptionAvailable()) {
    console.warn(
      "Secure OS storage is unavailable. " +
      "Google tokens will not be saved to disk."
    );

    try {
      if (fs.existsSync(GOOGLE_AUTH_FILE)) {
        fs.unlinkSync(GOOGLE_AUTH_FILE);
      }
    } catch (_) {}

    return;
  }

  const encrypted = safeStorage
    .encryptString(JSON.stringify(data))
    .toString("base64");

  atomicWriteJson(GOOGLE_AUTH_FILE, {
    encrypted,
  });
}

function clearGoogleAuthData() {
  if (fs.existsSync(GOOGLE_AUTH_FILE)) {
    fs.unlinkSync(GOOGLE_AUTH_FILE);
  }
  googleAuthData = null;
}

// ============================================================
// Google OAuth API
// ============================================================

function googleUserInfo(accessToken) {
  return httpsJsonRequest(
    "https://openidconnect.googleapis.com/v1/userinfo",
    {
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    }
  );
}

async function refreshGoogleAccessToken(data) {
  if (!data?.refreshToken || !GOOGLE_CLIENT_ID) {
    return null;
  }

  const body = new URLSearchParams({
    client_id: GOOGLE_CLIENT_ID,
    refresh_token: data.refreshToken,
    grant_type: "refresh_token",
  }).toString();

  const result = await httpsJsonRequest(
    "https://oauth2.googleapis.com/token",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
      },
      body,
    }
  );

  if (!result.access_token) {
    throw new Error(
      "Google did not return a refreshed access token."
    );
  }

  const next = {
    ...data,
    accessToken: result.access_token,
    expiresAt:
      Date.now() +
      (Number(result.expires_in) || 3600) * 1000,
    refreshToken:
      result.refresh_token || data.refreshToken,
  };

  saveGoogleAuthData(next);
  return next.accessToken;
}

async function getValidGoogleAccessToken() {
  const data = loadGoogleAuthData();

  if (!data) return null;

  if (
    data.accessToken &&
    Number(data.expiresAt) > Date.now() + 60000
  ) {
    return data.accessToken;
  }

  return refreshGoogleAccessToken(data);
}

function makePkcePair() {
  const verifier = crypto
    .randomBytes(32)
    .toString("base64url");

  const challenge = crypto
    .createHash("sha256")
    .update(verifier)
    .digest("base64url");

  return { verifier, challenge };
}

// ============================================================
// Google OAuth local callback
// ============================================================

function receiveGoogleOAuthCode(
  server,
  expectedState,
  timeoutMs = 180000
) {
  let timer = null;
  let settled = false;

  let resolveCode;
  let rejectCode;

  const promise = new Promise((resolve, reject) => {
    resolveCode = resolve;
    rejectCode = reject;
  });

  const finish = (error, code) => {
    if (settled) return;

    settled = true;

    if (timer) {
      clearTimeout(timer);
      timer = null;
    }

    if (error) rejectCode(error);
    else resolveCode(code);
  };

  server.on("request", (request, response) => {
    const remoteAddress =
      request.socket.remoteAddress || "";

    const localAddresses = [
      "127.0.0.1",
      "::1",
      "::ffff:127.0.0.1",
    ];

    if (!localAddresses.includes(remoteAddress)) {
      response.writeHead(403, {
        "Content-Type": "text/plain; charset=utf-8",
      });
      response.end("Forbidden");
      return;
    }

    const callbackUrl = new URL(
      request.url,
      "http://127.0.0.1"
    );

    if (callbackUrl.pathname !== "/") {
      response.writeHead(404, {
        "Content-Type": "text/plain; charset=utf-8",
      });
      response.end("Not found");
      return;
    }

    const params = callbackUrl.searchParams;

    if (params.get("state") !== expectedState) {
      response.writeHead(400, {
        "Content-Type": "text/plain; charset=utf-8",
      });

      response.end(
        "Sign-in state did not match. " +
        "You can close this tab and return to Sxarf AI."
      );

      finish(
        new Error(
          "Google sign-in failed the state validation check."
        )
      );

      return;
    }

    const oauthError = params.get("error");

    if (oauthError) {
      response.writeHead(400, {
        "Content-Type": "text/html; charset=utf-8",
      });

      response.end(
        "<!doctype html><title>Sign-in not completed</title>" +
        "<p>Google sign-in was not completed. " +
        "You can close this tab and return to Sxarf AI.</p>"
      );

      finish(
        new Error(
          `Google sign-in was not completed: ${oauthError}`
        )
      );

      return;
    }

    const code = params.get("code");

    if (!code) {
      response.writeHead(400, {
        "Content-Type": "text/plain; charset=utf-8",
      });

      response.end(
        "No authorization code was returned."
      );

      finish(
        new Error(
          "Google did not return an authorization code."
        )
      );

      return;
    }

    response.writeHead(200, {
      "Content-Type": "text/html; charset=utf-8",
      "Cache-Control": "no-store",
    });

    response.end(
      "<!doctype html><html><head><meta charset='utf-8'>" +
      "<title>Sxarf AI sign-in</title></head>" +
      "<body style='font:16px -apple-system,sans-serif;" +
      "padding:40px;background:#101112;color:#edf1f2'>" +
      "<h2>You're signed in</h2>" +
      "<p>Return to Sxarf AI. You can close this browser tab.</p>" +
      "</body></html>"
    );

    finish(null, code);
  });

  timer = setTimeout(() => {
    finish(
      new Error(
        "Google sign-in timed out. Please try again."
      )
    );
  }, timeoutMs);

  return {
    promise,
    cancel: error => {
      finish(
        error || new Error("Google sign-in cancelled.")
      );
    },
  };
}

// ============================================================
// Google sign-in flow
// ============================================================

async function performGoogleSignIn() {
  if (!GOOGLE_CLIENT_ID) {
    throw new Error(
      "Google sign-in is not configured. Set " +
      "SXARF_GOOGLE_CLIENT_ID to your Google Desktop OAuth " +
      "client ID, then restart Sxarf AI."
    );
  }

  if (googleSignInInProgress) {
    throw new Error(
      "A Google sign-in attempt is already in progress."
    );
  }

  googleSignInInProgress = true;

  const { verifier, challenge } = makePkcePair();
  const state = crypto.randomBytes(24).toString("hex");

  const server = http.createServer();
  let callback = null;

  try {
    await new Promise((resolve, reject) => {
      server.once("error", reject);
      server.listen(0, "127.0.0.1", resolve);
    });

    const address = server.address();

    if (!address || typeof address === "string") {
      throw new Error(
        "Could not open a local sign-in callback port."
      );
    }

    const redirectUri =
      `http://127.0.0.1:${address.port}`;

    callback = receiveGoogleOAuthCode(server, state);

    // Prevent an unhandled rejection if opening the browser fails.
    callback.promise.catch(() => {});

    const authUrl = new URL(
      "https://accounts.google.com/o/oauth2/v2/auth"
    );

    authUrl.search = new URLSearchParams({
      client_id: GOOGLE_CLIENT_ID,
      redirect_uri: redirectUri,
      response_type: "code",
      scope: "openid email profile",
      state,
      code_challenge: challenge,
      code_challenge_method: "S256",
      access_type: "offline",
      prompt: "consent",
    }).toString();

    await shell.openExternal(authUrl.toString());

    const code = await callback.promise;

    const tokenBody = new URLSearchParams({
      client_id: GOOGLE_CLIENT_ID,
      code,
      code_verifier: verifier,
      redirect_uri: redirectUri,
      grant_type: "authorization_code",
    }).toString();

    const tokens = await httpsJsonRequest(
      "https://oauth2.googleapis.com/token",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/x-www-form-urlencoded",
        },
        body: tokenBody,
      }
    );

    if (!tokens.access_token) {
      throw new Error(
        "Google did not return an access token."
      );
    }

    const profile = await googleUserInfo(
      tokens.access_token
    );

    const account = {
      id: String(profile.id || profile.sub || ""),
      name: String(
        profile.name ||
        profile.given_name ||
        "Google user"
      ),
      email: String(profile.email || ""),
      picture:
        typeof profile.picture === "string"
          ? profile.picture
          : "",
      provider: "google",
    };

    if (!account.email) {
      throw new Error(
        "Google did not return an email address. " +
        "Check the requested account scopes."
      );
    }

    const authData = {
      accessToken: tokens.access_token,
      refreshToken: tokens.refresh_token || null,
      expiresAt:
        Date.now() +
        (Number(tokens.expires_in) || 3600) * 1000,
      account,
    };

    saveGoogleAuthData(authData);

    return {
      account,
      persistentSession: Boolean(
        authData.refreshToken &&
        safeStorage.isEncryptionAvailable()
      ),
    };
  } finally {
    if (callback) {
      callback.cancel(
        new Error("Google sign-in flow ended.")
      );
    }

    if (server.listening) {
      server.close();
    }

    googleSignInInProgress = false;
  }
}

async function getGoogleAccount() {
  const accessToken = await getValidGoogleAccessToken();

  if (!accessToken) return null;

  const profile = await googleUserInfo(accessToken);

  const account = {
    id: String(profile.id || profile.sub || ""),
    name: String(
      profile.name ||
      profile.given_name ||
      "Google user"
    ),
    email: String(profile.email || ""),
    picture:
      typeof profile.picture === "string"
        ? profile.picture
        : "",
    provider: "google",
  };

  const current = loadGoogleAuthData() || {};

  saveGoogleAuthData({
    ...current,
    account,
  });

  return account;
}

// ============================================================
// Account and subscription IPC
// ============================================================

ipcMain.handle("account:auth-status", () => ({
  googleConfigured: Boolean(GOOGLE_CLIENT_ID),
  provider: "Google",
  message: GOOGLE_CLIENT_ID
    ? "Google OAuth client ID is configured."
    : "Set SXARF_GOOGLE_CLIENT_ID to enable Google sign-in.",
}));

ipcMain.handle("account:get", async () => {
  try {
    return (
      await getGoogleAccount() ||
      readLocalProfile()
    );
  } catch (error) {
    console.warn(
      "Could not restore Google account session:",
      error.message
    );

    return readLocalProfile();
  }
});

ipcMain.handle(
  "account:google-sign-in",
  async () => performGoogleSignIn()
);

ipcMain.handle("account:sign-out", () => {
  clearGoogleAuthData();
  return true;
});

ipcMain.handle("subscription:get", () => ({
  plan: readLocalPlan(),
  status: "local-test",
  billingConfigured: false,
  message:
    "Local visual testing only. " +
    "No payment or paid entitlement is granted.",
}));

ipcMain.handle(
  "local-profile:set",
  (_event, updates) => {
    if (!isPlainObject(updates)) {
      throw new Error("Invalid profile update.");
    }

    const current = readLocalProfile();
    const currentPlan = readLocalPlan();

    const name = typeof updates.name === "string"
      ? updates.name.trim().slice(0, 40)
      : current.name;

    const plan = typeof updates.plan === "string"
      ? updates.plan
      : currentPlan;

    if (!name) {
      throw new Error(
        "Display name cannot be empty."
      );
    }

    if (!ALLOWED_PLANS.includes(plan)) {
      throw new Error("Invalid plan.");
    }

    atomicWriteJson(LOCAL_PROFILE_FILE, {
      name,
      plan,
    });

    return {
      name,
      email: "",
      picture: "",
      provider: "local",
      plan,
    };
  }
);

ipcMain.handle(
  "subscription:checkout",
  (_event, plan) => {
    if (!ALLOWED_PLANS.includes(plan)) {
      throw new Error(
        "Unknown subscription plan."
      );
    }

    if (plan === "Free") {
      return {
        plan: "Free",
        status: "local",
      };
    }

    throw new Error(
      `${plan} checkout is not configured yet. ` +
      "No payment has been taken and no paid access has been granted."
    );
  }
);

// ============================================================
// CoCode: restricted project filesystem access
// ============================================================

let cocodeRoot = null;

const COCODE_MAX_FILE_SIZE = 300 * 1024;
const COCODE_MAX_WRITE_SIZE = 500 * 1024;
const COCODE_MAX_FILES = 250;
const COCODE_MAX_DEPTH = 8;

const COCODE_IGNORED_DIRS = new Set([
  ".git",
  ".svn",
  ".hg",
  "node_modules",
  "venv",
  ".venv",
  "__pycache__",
  ".next",
  "dist",
  "build",
  "target",
  "vendor",
  ".idea",
  ".vscode",
  "coverage",
  ".cache",
]);

const COCODE_BLOCKED_NAMES = new Set([
  ".env",
  ".env.local",
  ".env.production",
  ".npmrc",
  ".pypirc",
  "id_rsa",
  "id_ed25519",
  "credentials",
  "secrets",
  "service-account.json",
]);

const COCODE_ALLOWED_EXTENSIONS = new Set([
  ".txt",
  ".md",
  ".markdown",
  ".js",
  ".jsx",
  ".ts",
  ".tsx",
  ".mjs",
  ".cjs",
  ".py",
  ".pyw",
  ".html",
  ".htm",
  ".css",
  ".scss",
  ".json",
  ".jsonc",
  ".yaml",
  ".yml",
  ".toml",
  ".ini",
  ".xml",
  ".svg",
  ".sh",
  ".bash",
  ".zsh",
  ".c",
  ".h",
  ".cpp",
  ".hpp",
  ".java",
  ".kt",
  ".go",
  ".rs",
  ".swift",
  ".rb",
  ".php",
  ".sql",
  ".lua",
  ".gitignore",
  ".gitattributes",
  ".dockerfile",
]);

function cocodeRelativePath(input) {
  if (
    typeof input !== "string" ||
    !input.trim()
  ) {
    throw new Error(
      "Invalid project file path."
    );
  }

  const normalized = input.replace(/\\/g, "/");

  if (
    normalized.startsWith("/") ||
    /^[a-zA-Z]:/.test(normalized) ||
    normalized.includes("\0")
  ) {
    throw new Error(
      "Absolute paths are not permitted."
    );
  }

  const parts = normalized.split("/");

  if (
    parts.some(part =>
      !part ||
      part === "." ||
      part === ".." ||
      part.startsWith(".")
    )
  ) {
    throw new Error(
      "Hidden files and path traversal are blocked."
    );
  }

  if (
    parts.some(part =>
      COCODE_BLOCKED_NAMES.has(
        part.toLowerCase()
      )
    )
  ) {
    throw new Error(
      "This file is protected."
    );
  }

  return parts.join("/");
}

function cocodeIsAllowedFile(relative) {
  const base = relative
    .split("/")
    .pop()
    .toLowerCase();

  if (COCODE_BLOCKED_NAMES.has(base)) {
    return false;
  }

  if (
    base.startsWith(".env") ||
    [".pem", ".key", ".p12", ".pfx"].some(
      extension => base.endsWith(extension)
    )
  ) {
    return false;
  }

  return COCODE_ALLOWED_EXTENSIONS.has(
    path.extname(base)
  );
}

function cocodeGetRoot() {
  if (!cocodeRoot) {
    throw new Error(
      "No project folder is connected."
    );
  }

  try {
    const stat = fs.statSync(cocodeRoot);

    if (!stat.isDirectory()) {
      throw new Error("Not a directory.");
    }

    return fs.realpathSync(cocodeRoot);
  } catch (_) {
    cocodeRoot = null;

    throw new Error(
      "The connected project folder is unavailable."
    );
  }
}

function cocodeResolve(input, allowMissing = false) {
  const relative = cocodeRelativePath(input);
  const root = cocodeGetRoot();

  const candidate = path.resolve(
    root,
    ...relative.split("/")
  );

  if (
    candidate !== root &&
    !candidate.startsWith(root + path.sep)
  ) {
    throw new Error(
      "Path escapes the connected project."
    );
  }

  let current = root;

  for (const part of relative.split("/")) {
    current = path.join(current, part);

    if (!fs.existsSync(current)) {
      if (allowMissing) continue;

      throw new Error(
        "Project file does not exist."
      );
    }

    const stat = fs.lstatSync(current);

    if (stat.isSymbolicLink()) {
      throw new Error(
        "Symbolic links are not permitted."
      );
    }

    const real = fs.realpathSync(current);

    if (
      real !== root &&
      !real.startsWith(root + path.sep)
    ) {
      throw new Error(
        "Path escapes the connected project."
      );
    }
  }

  return {
    relative,
    absolute: candidate,
  };
}

async function cocodeChooseDirectory() {
  return dialog.showOpenDialog(
    mainWindow || undefined,
    {
      title: "Choose a project folder for CoCode",
      buttonLabel: "Connect project",
      properties: ["openDirectory"],
    }
  );
}

ipcMain.handle(
  "cocode:choose-directory",
  async () => {
    const result = await cocodeChooseDirectory();

    if (
      result.canceled ||
      !result.filePaths.length
    ) {
      return {
        connected: false,
        cancelled: true,
      };
    }

    const selected = result.filePaths[0];

    try {
      const real = fs.realpathSync(selected);

      if (!fs.statSync(real).isDirectory()) {
        throw new Error(
          "Please select a folder."
        );
      }

      cocodeRoot = real;

      return {
        connected: true,
        name: path.basename(real),
      };
    } catch (error) {
      cocodeRoot = null;

      throw new Error(
        "Could not connect that project folder: " +
        error.message
      );
    }
  }
);

ipcMain.handle(
  "cocode:get-directory",
  async () => {
    if (!cocodeRoot) {
      return { connected: false };
    }

    try {
      cocodeGetRoot();

      return {
        connected: true,
        name: path.basename(cocodeRoot),
      };
    } catch (_) {
      cocodeRoot = null;
      return { connected: false };
    }
  }
);

ipcMain.handle(
  "cocode:disconnect",
  async () => {
    cocodeRoot = null;
    return { connected: false };
  }
);

ipcMain.handle(
  "cocode:list-files",
  async (_event, options = {}) => {
    const root = cocodeGetRoot();

    const requestedMax = Number(options?.maxFiles);

    const maxFiles = Number.isFinite(requestedMax)
      ? Math.max(
          1,
          Math.min(
            Math.floor(requestedMax),
            COCODE_MAX_FILES
          )
        )
      : 180;

    const output = [];

    function walk(directory, depth) {
      if (
        depth > COCODE_MAX_DEPTH ||
        output.length >= maxFiles
      ) {
        return;
      }

      let entries;

      try {
        entries = fs.readdirSync(directory, {
          withFileTypes: true,
        });
      } catch (_) {
        return;
      }

      for (const entry of entries) {
        if (output.length >= maxFiles) break;

        const name = entry.name;
        const lowerName = name.toLowerCase();

        if (
          name.startsWith(".") ||
          COCODE_IGNORED_DIRS.has(lowerName) ||
          COCODE_BLOCKED_NAMES.has(lowerName)
        ) {
          continue;
        }

        const absolute = path.join(
          directory,
          name
        );

        const relative = path
          .relative(root, absolute)
          .split(path.sep)
          .join("/");

        try {
          const stat = fs.lstatSync(absolute);

          if (stat.isSymbolicLink()) continue;

          if (stat.isDirectory()) {
            walk(absolute, depth + 1);
            continue;
          }

          if (!stat.isFile()) continue;
          if (stat.size > COCODE_MAX_FILE_SIZE) continue;
          if (!cocodeIsAllowedFile(relative)) continue;

          output.push({
            path: relative,
            size: stat.size,
          });
        } catch (_) {
          // Ignore inaccessible files.
        }
      }
    }

    walk(root, 0);
    return output;
  }
);

ipcMain.handle(
  "cocode:read-file",
  async (_event, input) => {
    const resolved = cocodeResolve(input);
    const stat = fs.lstatSync(resolved.absolute);

    if (
      stat.isSymbolicLink() ||
      !stat.isFile()
    ) {
      throw new Error(
        "Not a regular file."
      );
    }

    if (stat.size > COCODE_MAX_FILE_SIZE) {
      throw new Error(
        "File is too large to read."
      );
    }

    if (!cocodeIsAllowedFile(resolved.relative)) {
      throw new Error(
        "This file type is not permitted."
      );
    }

    const content = fs.readFileSync(
      resolved.absolute
    );

    if (content.includes(0)) {
      throw new Error(
        "Binary files are not supported."
      );
    }

    return content.toString("utf8");
  }
);

ipcMain.handle(
  "cocode:write-file",
  async (_event, change) => {
    if (!isPlainObject(change)) {
      throw new Error("Invalid file change.");
    }

    if (typeof change.content !== "string") {
      throw new Error(
        "File content must be text."
      );
    }

    if (
      Buffer.byteLength(change.content, "utf8") >
      COCODE_MAX_WRITE_SIZE
    ) {
      throw new Error(
        "Proposed file is too large."
      );
    }

    const resolved = cocodeResolve(
      change.path,
      true
    );

    if (!cocodeIsAllowedFile(resolved.relative)) {
      throw new Error(
        "This file type is not permitted."
      );
    }

    const root = cocodeGetRoot();
    const parent = path.dirname(resolved.absolute);

    if (
      !fs.existsSync(parent) ||
      !fs.statSync(parent).isDirectory()
    ) {
      throw new Error(
        "The parent directory must already exist."
      );
    }

    const parentReal = fs.realpathSync(parent);

    if (
      parentReal !== root &&
      !parentReal.startsWith(root + path.sep)
    ) {
      throw new Error(
        "The destination escapes the project folder."
      );
    }

    if (fs.existsSync(resolved.absolute)) {
      const stat = fs.lstatSync(resolved.absolute);

      if (
        stat.isSymbolicLink() ||
        !stat.isFile()
      ) {
        throw new Error(
          "The destination is not a safe regular file."
        );
      }
    }

    const temporary = path.join(
      parent,
      ".cocode-" +
      crypto.randomBytes(12).toString("hex") +
      ".tmp"
    );

    try {
      fs.writeFileSync(
        temporary,
        change.content,
        {
          encoding: "utf8",
          flag: "wx",
          mode: 0o600,
        }
      );

      fs.renameSync(
        temporary,
        resolved.absolute
      );
    } catch (error) {
      try {
        if (fs.existsSync(temporary)) {
          fs.unlinkSync(temporary);
        }
      } catch (_) {}

      throw error;
    }

    return {
      written: true,
      path: resolved.relative,
    };
  }
);

// ============================================================
// MLX backend health checking
// ============================================================

function checkServerReady(maxWaitMs = 600000) {
  const startedAt = Date.now();
  const pollIntervalMs = 500;
  const requestTimeoutMs = 2500;

  return new Promise((resolve, reject) => {
    let settled = false;
    let interval = null;
    let activeRequest = null;

    function finish(error) {
      if (settled) return;

      settled = true;

      if (interval) {
        clearInterval(interval);
        interval = null;
      }

      if (activeRequest) {
        activeRequest.destroy();
        activeRequest = null;
      }

      if (error) reject(error);
      else resolve(true);
    }

    function poll() {
      if (settled) return;

      if (backendSpawnError) {
        finish(
          new Error(
            `Could not launch the ${IS_WINDOWS ? "Windows CPU" : "MLX"} backend: ` +
            backendSpawnError.message
          )
        );
        return;
      }

      if (backendExit) {
        finish(
          new Error(
            `${IS_WINDOWS ? "Windows CPU" : "MLX"} server exited (code=${backendExit.code}, ` +
            `signal=${backendExit.signal}).`
          )
        );
        return;
      }

      if (Date.now() - startedAt >= maxWaitMs) {
        finish(
          new Error(
            `${IS_WINDOWS ? "Windows CPU" : "MLX"} server did not become ready ` +
            "within the startup timeout."
          )
        );
        return;
      }

      if (activeRequest) return;

      const request = http.get(
        `${SERVER_URL}/health`,
        response => {
          let body = "";

          response.setEncoding("utf8");

          response.on("data", chunk => {
            body += chunk;
          });

          response.on("end", () => {
            if (settled) return;

            activeRequest = null;

            if (response.statusCode !== 200) {
              return;
            }

            try {
              const health = JSON.parse(body);

              if (health.status === "ready") {
                finish();
              } else if (health.status === "error") {
                finish(
                  new Error(
                    health.error ||
                    "The default local AI model failed to load."
                  )
                );
              }
            } catch (_) {
              // The backend may still be starting.
            }
          });
        }
      );

      activeRequest = request;

      request.setTimeout(
        requestTimeoutMs,
        () => {
          request.destroy(
            new Error("Health request timed out.")
          );
        }
      );

      request.on("error", () => {
        if (activeRequest === request) {
          activeRequest = null;
        }
      });
    }

    interval = setInterval(
      poll,
      pollIntervalMs
    );

    poll();
  });
}

// ============================================================
// MLX backend process
// ============================================================

function startPythonServer() {
  if (!fs.existsSync(SERVER_SCRIPT)) {
    throw new Error(
      `Backend script not found: ${SERVER_SCRIPT}`
    );
  }

  if (!fs.existsSync(VENV_PYTHON)) {
    throw new Error(
      `Python environment not found: ${VENV_PYTHON}`
    );
  }

  if (pythonProcess) return;

  backendExit = null;
  backendSpawnError = null;

  console.log(
    `Starting ${IS_WINDOWS ? "Windows CPU" : "MLX"} backend with ${VENV_PYTHON}`
  );

  const backendEnv = {
    ...process.env,
    HF_HOME,
    SXARF_USER_DATA: SUPPORT_DIR,
    HUGGINGFACE_HUB_CACHE: path.join(HF_HOME, "hub"),
    PYTHONUNBUFFERED: "1",
    SXARF_SUPPORT_DIR: SUPPORT_DIR,
  };
  if (!IS_WINDOWS) {
    backendEnv.DEFAULT_MODEL = "llama32";
    backendEnv.PATH = [
      "/opt/homebrew/bin",
      "/usr/local/bin",
      process.env.PATH || "",
    ].join(path.delimiter);
  }

  pythonProcess = spawn(
    VENV_PYTHON,
    [
      SERVER_SCRIPT,
      "--host",
      "127.0.0.1",
      "--port",
      String(SERVER_PORT),
    ],
    {
      // Use the actual server directory in both dev and packaged builds.
      cwd: path.dirname(SERVER_SCRIPT),

      stdio: [
        "ignore",
        "pipe",
        "pipe",
      ],
      env: backendEnv,
    }
  );

  const child = pythonProcess;

  child.stdout.on("data", data => {
    console.log(
      `[${IS_WINDOWS ? "CPU" : "MLX"}] ${data.toString().trimEnd()}`
    );
  });

  child.stderr.on("data", data => {
    console.error(
      `[${IS_WINDOWS ? "CPU" : "MLX"}] ${data.toString().trimEnd()}`
    );
  });

  child.on("error", error => {
    backendSpawnError = error;

    console.error(
      `Could not launch ${IS_WINDOWS ? "Windows CPU" : "MLX"} backend:`,
      error
    );
  });

  child.on("exit", (code, signal) => {
    console.log(
      `${IS_WINDOWS ? "Windows CPU" : "MLX"} backend exited (code=${code}, signal=${signal})`
    );

    backendExit = {
      code,
      signal,
    };

    if (pythonProcess === child) {
      pythonProcess = null;
    }

    if (
      mainWindow &&
      !mainWindow.isDestroyed() &&
      !quitting
    ) {
      mainWindow.webContents.send(
        "backend:stopped",
        {
          code,
          signal,
        }
      );
    }
  });
}

// ============================================================
// First-launch setup
// ============================================================

function reportSetupProgress(message) {
  setupProgressText = message;
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send("setup:progress", message);
  }
}

ipcMain.handle("setup:get-progress", () => setupProgressText);

function findPythonExecutable() {
  const candidates = IS_WINDOWS
    ? [
        { command: "py", args: ["-3.12"] },
        { command: "py", args: ["-3.11"] },
        { command: "py", args: ["-3.10"] },
        { command: "python", args: [] },
      ]
    : [
        "python3.13", "python3.12", "python3.11", "python3.10",
        "/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3",
        "python3",
      ].map(command => ({ command, args: [] }));

  for (const candidate of candidates) {
    try {
      const version = execFileSync(candidate.command, [
        ...candidate.args,
        "-c",
        "import platform,sys; print(f'{sys.version_info.major}.{sys.version_info.minor} {platform.machine()}')",
      ], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
        env: {
          ...process.env,
          PATH: IS_WINDOWS
            ? process.env.PATH || ""
            : ["/opt/homebrew/bin", "/usr/local/bin", process.env.PATH || ""].join(path.delimiter),
        },
      }).trim();
      const [versionText, architecture] = version.split(/\s+/);
      const [major, minor] = versionText.split(".").map(Number);
      const supportedArchitecture = IS_WINDOWS
        ? ["amd64", "x86_64"].includes(architecture.toLowerCase())
        : architecture === "arm64";
      const supportedPythonVersion = IS_WINDOWS
        ? major === 3 && minor >= 10 && minor <= 12
        : major > 3 || (major === 3 && minor >= 10);
      if (supportedPythonVersion && supportedArchitecture) {
        return candidate;
      }
    } catch (_) {
      // Try the next installed Python executable.
    }
  }

  throw new Error(
    IS_WINDOWS
      ? "64-bit Python 3.10–3.12 was not found. Install Python from python.org, enable the Python launcher, then reopen Sxarf AI."
      : "Python 3.10 or newer for Apple Silicon was not found. Install Python 3 from python.org or Homebrew, then reopen Sxarf AI."
  );
}

function runBootstrapCommand(command, args, label, env) {
  return new Promise((resolve, reject) => {
    reportSetupProgress(label);
    let output = "";
    let lineBuffer = "";
    const child = spawn(command, args, {
      cwd: APP_RESOURCES,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    bootstrapProcess = child;

    const collect = (chunk, isError) => {
      const text = chunk.toString();
      output = (output + text).slice(-8000);
      if (isError) console.error(`[Setup] ${text.trimEnd()}`);
      else console.log(`[Setup] ${text.trimEnd()}`);
      lineBuffer += text;
      const lines = lineBuffer.split(/[\r\n]+/);
      lineBuffer = lines.pop() || "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (trimmed) reportSetupProgress(trimmed.slice(-180));
      }
    };

    child.stdout.on("data", chunk => collect(chunk, false));
    child.stderr.on("data", chunk => collect(chunk, true));
    child.once("error", error => {
      if (bootstrapProcess === child) bootstrapProcess = null;
      reject(new Error(`${label} Could not start: ${error.message}`));
    });
    child.once("close", (code, signal) => {
      if (bootstrapProcess === child) bootstrapProcess = null;
      if (lineBuffer.trim()) reportSetupProgress(lineBuffer.trim().slice(-180));
      if (code === 0) resolve();
      else reject(new Error(
        `${label} failed (exit code ${code ?? "unknown"}, signal ${signal || "none"}).` +
        (output.trim() ? `\n\n${output.trim()}` : "")
      ));
    });
  });
}

async function runFirstLaunchSetup() {
  fs.mkdirSync(SUPPORT_DIR, { recursive: true });
  fs.mkdirSync(HF_HOME, { recursive: true });

  if (!fs.existsSync(REQUIREMENTS_FILE)) {
    throw new Error(`Bundled Python dependencies file not found: ${REQUIREMENTS_FILE}`);
  }
  if (!fs.existsSync(SERVER_SCRIPT)) {
    throw new Error(`Bundled Python backend not found: ${SERVER_SCRIPT}`);
  }

  const requirementsHash = crypto
    .createHash("sha256")
    .update(fs.readFileSync(REQUIREMENTS_FILE))
    .digest("hex");
  let setupIsComplete = false;
  if (fs.existsSync(SETUP_MARKER) && fs.existsSync(VENV_PYTHON)) {
    try {
      setupIsComplete = fs.readFileSync(SETUP_MARKER, "utf8").trim() === requirementsHash;
      if (setupIsComplete) {
        const requiredPackages = IS_WINDOWS
          ? "llama_cpp,fastapi,uvicorn"
          : "mlx_lm,mlx_whisper,fastapi,uvicorn";
        execFileSync(VENV_PYTHON, [
          "-c",
          `import importlib.util,sys; sys.exit(any(importlib.util.find_spec(p) is None for p in '${requiredPackages}'.split(',')))`,
        ], {
          stdio: "ignore",
        });
      }
    } catch (_) {
      setupIsComplete = false;
    }
  }
  if (setupIsComplete) {
    reportSetupProgress("Local Python environment is ready.");
    return;
  }

  const setupEnv = {
    ...process.env,
    SXARF_USER_DATA: SUPPORT_DIR,
    SXARF_SUPPORT_DIR: SUPPORT_DIR,
    HF_HOME,
    HUGGINGFACE_HUB_CACHE: path.join(HF_HOME, "hub"),
    PATH: IS_WINDOWS
      ? process.env.PATH || ""
      : ["/opt/homebrew/bin", "/usr/local/bin", process.env.PATH || ""].join(path.delimiter),
  };
  const python = findPythonExecutable();

  if (!fs.existsSync(VENV_PYTHON)) {
    reportSetupProgress("Setting up local Python environment...");
    if (fs.existsSync(VENV_DIR)) {
      const current = fs.lstatSync(VENV_DIR);
      if (current.isSymbolicLink() || !current.isDirectory()) {
        throw new Error(`Unsafe Python environment path: ${VENV_DIR}`);
      }
      fs.rmSync(VENV_DIR, { recursive: true, force: true });
    }
    await runBootstrapCommand(
      python.command,
      [...python.args, "-m", "venv", VENV_DIR],
      "Creating Python virtual environment...",
      setupEnv
    );
  }

  if (!fs.existsSync(VENV_PYTHON)) {
    throw new Error(`Python virtual environment was not created at ${VENV_DIR}`);
  }

  const pipArgs = [
    "-m",
    "pip",
    "install",
    "--disable-pip-version-check",
  ];
  if (IS_WINDOWS) {
    pipArgs.push(
      "--extra-index-url",
      "https://abetlen.github.io/llama-cpp-python/whl/cpu"
    );
  }
  pipArgs.push("-r", REQUIREMENTS_FILE);

  await runBootstrapCommand(
    VENV_PYTHON,
    pipArgs,
    "Installing AI dependencies...",
    setupEnv
  );

  const requiredPackages = IS_WINDOWS
    ? "llama_cpp,fastapi,uvicorn"
    : "mlx_lm,mlx_whisper,fastapi,uvicorn";
  execFileSync(VENV_PYTHON, [
    "-c",
    `import importlib.util,sys; required='${requiredPackages}'.split(','); missing=[p for p in required if importlib.util.find_spec(p) is None]; print(', '.join(missing)); sys.exit(bool(missing))`,
  ], {
    stdio: "inherit",
    env: setupEnv,
  });
  fs.writeFileSync(SETUP_MARKER, `${requirementsHash}\n`, { mode: 0o600 });
  reportSetupProgress("AI dependencies installed. Starting local model...");
}

// ============================================================
// Backend shutdown
// ============================================================

function stopPythonServer() {
  if (!pythonProcess) return;

  const child = pythonProcess;

  if (
    child.exitCode !== null ||
    child.signalCode !== null
  ) {
    return;
  }

  try {
    child.kill(IS_WINDOWS ? undefined : "SIGTERM");
  } catch (error) {
    console.warn(
    `Could not stop ${IS_WINDOWS ? "Windows CPU" : "MLX"} backend gracefully:`,
      error
    );
  }

  const forceKillTimer = setTimeout(() => {
    if (
      child.exitCode === null &&
      child.signalCode === null
    ) {
      try {
        child.kill();
      } catch (error) {
        console.warn(
            `Could not force-stop ${IS_WINDOWS ? "Windows CPU" : "MLX"} backend:`,
          error
        );
      }
    }
  }, 2500);

  forceKillTimer.unref();
}

// ============================================================
// Browser window
// ============================================================

function createWindow() {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.focus();
    return;
  }

  mainWindow = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 760,
    minHeight: 560,
    title: "Sxarf AI",
    backgroundColor: "#0b0b0b",
    show: false,

    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.loadFile(
    path.join(__dirname, "index.html")
  );

  mainWindow.once("ready-to-show", () => {
    if (
      mainWindow &&
      !mainWindow.isDestroyed()
    ) {
      mainWindow.show();
    }
  });

  mainWindow.webContents.setWindowOpenHandler(
    ({ url }) => {
      if (
        url.startsWith("https://") ||
        url.startsWith("http://")
      ) {
        shell.openExternal(url).catch(error => {
          console.error(
            "Could not open external link:",
            error
          );
        });
      }

      return {
        action: "deny",
      };
    }
  );

  mainWindow.webContents.on(
    "will-navigate",
    (event, url) => {
      if (!url.startsWith("file://")) {
        event.preventDefault();

        if (
          url.startsWith("https://") ||
          url.startsWith("http://")
        ) {
          shell.openExternal(url).catch(error => {
            console.error(
              "Could not open external link:",
              error
            );
          });
        }
      }
    }
  );

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

// ============================================================
// Application menu
// ============================================================

const menuTemplate = [
  {
    label: "File",
    submenu: [
      {
        role: "quit",
      },
    ],
  },

  {
    label: "Edit",
    submenu: [
      { role: "undo" },
      { role: "redo" },
      { type: "separator" },
      { role: "cut" },
      { role: "copy" },
      { role: "paste" },
      { role: "selectAll" },
    ],
  },

  {
    label: "View",
    submenu: [
      { role: "reload" },
      { role: "forceReload" },
      { role: "toggleDevTools" },
      { type: "separator" },
      { role: "togglefullscreen" },
    ],
  },
];

// ============================================================
// Application startup
// ============================================================

async function startApplication() {
  Menu.setApplicationMenu(
    Menu.buildFromTemplate(menuTemplate)
  );

  const supportedHost =
    (process.platform === "darwin" && process.arch === "arm64") ||
    (process.platform === "win32" && process.arch === "x64");
  if (!supportedHost) {
    dialog.showErrorBox(
      "Unsupported computer",
      "The macOS build requires an Apple Silicon Mac. " +
      "The Windows build requires 64-bit Windows."
    );

    app.quit();
    return;
  }

  try {
    createWindow();
    await runFirstLaunchSetup();

    reportSetupProgress("Starting the local AI backend...");
    startPythonServer();

    await checkServerReady(IS_WINDOWS ? 1800000 : 600000);

    if (quitting) return;
    reportSetupProgress("Local AI model is ready.");
  } catch (error) {
    console.error(
      "Sxarf AI startup failed:",
      error
    );

    dialog.showErrorBox(
      `${IS_WINDOWS ? "Windows CPU" : "MLX"} Backend Error`,
      `${error.message}\n\n` +
      (IS_WINDOWS
        ? "Check the first-launch setup log, install 64-bit Python 3.10–3.12, and confirm the local model can be downloaded."
        : "Check the first-launch setup log and confirm that the runtime contains mlx-lm, mlx-vlm, mlx-whisper, FastAPI, and Uvicorn.")
    );

    quitting = true;
    stopPythonServer();
    app.quit();
  }
}

// ============================================================
// Electron lifecycle
// ============================================================

app.whenReady().then(() => {
  session.defaultSession.setPermissionRequestHandler((webContents, permission, callback, details) => {
    const isAppWindow = webContents === mainWindow?.webContents;
    const requestsAudio = permission === "media" &&
      Array.isArray(details?.mediaTypes) &&
      details.mediaTypes.includes("audio") &&
      !details.mediaTypes.includes("video");
    if (!isAppWindow || !requestsAudio) {
      callback(false);
      return;
    }

    if (process.platform === "darwin") {
      systemPreferences.askForMediaAccess("microphone")
        .then(callback)
        .catch(error => {
          console.error("Could not request microphone access:", error);
          callback(false);
        });
      return;
    }
    callback(true);
  });

  // Prevent accidental duplicate startup.
  if (!startupPromise) {
    startupPromise = startApplication();
  }

  app.on("activate", () => {
    if ( 
      BrowserWindow.getAllWindows().length === 0 &&
      !quitting
    ) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  // Keep the backend alive on macOS so the Dock can reopen the app.
  if (process.platform !== "darwin") {
    stopPythonServer();
    app.quit();
  }
});

app.on("before-quit", () => {
  quitting = true;
  stopVoiceProcess();
  if (bootstrapProcess && bootstrapProcess.exitCode === null) {
    bootstrapProcess.kill();
    bootstrapProcess = null;
  }
  stopPythonServer();
});