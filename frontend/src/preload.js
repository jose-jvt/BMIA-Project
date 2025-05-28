// See the Electron documentation for details on how to use preload scripts:
// https://www.electronjs.org/docs/latest/tutorial/process-model#preload-scripts

const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  selectFiles: () => ipcRenderer.invoke('dialog:selectFiles'),
  selectModels: () => ipcRenderer.invoke('dialog:selectModels'),
  selectMRIs: () => ipcRenderer.invoke('dialog:selectMRIs'),
  selectDSIStudioPath: () => ipcRenderer.invoke('dialog:openDSIStudioPath')
});
