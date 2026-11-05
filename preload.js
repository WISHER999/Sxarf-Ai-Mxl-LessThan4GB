const { contextBridge, ipcRenderer, clipboard } = require("electron");

contextBridge.exposeInMainWorld("electronAPI", {
  // Local app state (chats, memory, and UI preferences).
  getAppState: () => ipcRenderer.invoke("app-state:load"),
  saveAppState: state => ipcRenderer.invoke("app-state:save", state),
  setLocalProfile: updates => ipcRenderer.invoke("local-profile:set", updates),

  // Google OAuth is performed in the Electron main process, never in the page.
  getAuthStatus: () => ipcRenderer.invoke("account:auth-status"),
  getAccount: () => ipcRenderer.invoke("account:get"),
  signInWithGoogle: () => ipcRenderer.invoke("account:google-sign-in"),
  signOut: () => ipcRenderer.invoke("account:sign-out"),

  // Billing remains explicitly disabled until a real checkout/entitlement service exists.
  getSubscription: () => ipcRenderer.invoke("subscription:get"),
  startCheckout: plan => ipcRenderer.invoke("subscription:checkout", plan),

  copyText: text => {
    if (typeof text !== "string") return Promise.reject(new TypeError("Text to copy must be a string."));
    clipboard.writeText(text);
    return Promise.resolve(true);
  },

  speakText: text => ipcRenderer.invoke("voice:speak", text),
  stopSpeaking: () => ipcRenderer.invoke("voice:stop-speaking"),
  onSetupProgress: callback => {
    if (typeof callback !== "function") return () => {};
    const listener = (_event, message) => callback(message);
    ipcRenderer.on("setup:progress", listener);
    ipcRenderer.invoke("setup:get-progress").then(callback).catch(error => {
      console.error("Could not read setup progress:", error);
    });
    return () => ipcRenderer.removeListener("setup:progress", listener);
  },

  // CoCode: narrow project-folder operations.
  cocodeChooseDirectory: () => ipcRenderer.invoke("cocode:choose-directory"),
  cocodeGetDirectory: () => ipcRenderer.invoke("cocode:get-directory"),
  cocodeDisconnect: () => ipcRenderer.invoke("cocode:disconnect"),
  cocodeListFiles: options => ipcRenderer.invoke("cocode:list-files", options),
  cocodeReadFile: relativePath => ipcRenderer.invoke("cocode:read-file", relativePath),
  cocodeWriteFile: change => ipcRenderer.invoke("cocode:write-file", change),

onBackendStopped: callback => {
    if (typeof callback !== "function") return () => {};
    const listener = (_event, details) => callback(details);
    ipcRenderer.on("backend:stopped", listener);
    return () => ipcRenderer.removeListener("backend:stopped", listener);
  },
});
