const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('node:path');

// Handle creating/removing shortcuts on Windows when installing/uninstalling.
if (require('electron-squirrel-startup')) {
  app.quit();
}

const createWindow = () => {
  const mainWindow = new BrowserWindow({
    width: 1270,
    height: 800,
    minWidth: 1270,
    minHeight: 800,
    maxWidth: 1270,
    maxHeight: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));
  // mainWindow.webContents.openDevTools(); // Optional
};

app.whenReady().then(() => {
  createWindow();

  // Register IPC only after Electron is ready
  ipcMain.handle('dialog:openDSIStudioPath', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
      title: 'Select DSI Studio Executable',
      properties: ['openFile'],
      filters: [{ name: 'Executables', extensions: process.platform === 'win32' ? ['exe'] : ['app'] }]
    });

    return canceled ? null : filePaths[0];
  });

  ipcMain.handle('dialog:selectFiles', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
      title: 'Select Connectivity Matrices',
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'Supported Files', extensions: ['mat'] }]
    });

    return canceled ? [] : filePaths; // Return empty array if user cancels
  });

  ipcMain.handle('dialog:selectModels', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
      title: 'Select PyTorch models',
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'Supported Files', extensions: ['pth'] }]
    });

    return canceled ? [] : filePaths; // Return empty array if user cancels
  });

  ipcMain.handle('dialog:selectMRIs', async () => {
    const { canceled, filePaths } = await dialog.showOpenDialog({
      title: 'Select PyTorch models',
      properties: ['openFile', 'multiSelections'],
      filters: [{ name: 'Supported Files', extensions: ['nii', 'nii.gz', 'dcm', 'fdf'] }]
    });

    return canceled ? [] : filePaths; // Return empty array if user cancels
  });


  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
