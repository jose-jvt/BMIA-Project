// Global variables
let selectedFilePaths = [];
let selectedModelPaths = [];
let selectedExportPaths = [];
let selectedFiles = new Set(); // Track selected dMRI files
let selectedModels = new Set(); // Track selected model files
let fileCounter = 0;
let modelCounter = 0;

// Modal state variables
let pendingNiftiFiles = []; // Files waiting for modal completion
let currentModalFileIndex = 0;

// DOM Elements
const elements = {
    // Section 1 (dMRI)
    uploadBtn: document.getElementById('uploadBtn'),
    fileInput: document.getElementById('fileInput'),
    fileList: document.getElementById('fileList'),
    filesContainer: document.getElementById('filesContainer'),
    clearAllBtn: document.getElementById('clearAllBtn'),
    uploadAndExportBtn: document.getElementById('uploadAndExportBtn'),
    
    // Section 2 (PyTorch)
    uploadBtn2: document.getElementById('uploadBtn2'),
    fileInput2: document.getElementById('fileInput2'),
    fileList2: document.getElementById('fileList2'),
    filesContainer2: document.getElementById('filesContainer2'),
    clearAllBtn2: document.getElementById('clearAllBtn2'),
    
    // Controls
    processingMethod: document.getElementById('processingMethod'),
    dsiPathDisplay: document.getElementById('dsiPathDisplay'),
    runBtn: document.getElementById('runBtn'),
    resultsOutput: document.getElementById('resultsOutput'),
    
    // Modal elements
    niftiModal: document.getElementById('niftiModal'),
    modalFilename: document.getElementById('modalFilename'),
    niftiItem1: document.getElementById('niftiItem1'),
    niftiItem2: document.getElementById('niftiItem2'),
    modalCancel: document.getElementById('modalCancel'),
    modalSubmit: document.getElementById('modalSubmit')
};

// Modal functions
function showNiftiModal(filename) {
    elements.modalFilename.textContent = filename;
    elements.niftiItem1.value = '';
    elements.niftiItem2.value = '';
    elements.niftiModal.classList.add('show');
    elements.niftiItem1.focus();
}

function hideNiftiModal() {
    elements.niftiModal.classList.remove('show');
    elements.modalFilename.textContent = '';
    elements.niftiItem1.value = '';
    elements.niftiItem2.value = '';
}

function processNextNiftiFile() {
    if (currentModalFileIndex < pendingNiftiFiles.length) {
        const file = pendingNiftiFiles[currentModalFileIndex];
        showNiftiModal(file.name);
    } else {
        // All files processed, finalize the export process
        finalizeExportProcess();
    }
}

function finalizeExportProcess() {
    // Reset modal state
    pendingNiftiFiles = [];
    currentModalFileIndex = 0;
    
    // Add all processed files to the main container
    selectedExportPaths.forEach(fileData => {
        createFileItem(fileData.id, fileData.name, fileData.path, 'removeFile', elements.filesContainer);
    });
    
    // Update display
    toggleFileDisplay(elements.clearAllBtn, selectedFilePaths.length > 0 || selectedExportPaths.length > 0);
    showExportResults();
}

function showExportResults() {
    // Display export execution results
    elements.resultsOutput.value = '=== EXPORT TO CONNECTIVITY MATRIX EXECUTION ===\n\n';
    elements.resultsOutput.value += '📁 PROCESSED FILES FOR EXPORT:\n';
    
    selectedExportPaths.forEach((fileData, index) => {
        elements.resultsOutput.value += `${index + 1}. ${fileData.name}\n`;
        elements.resultsOutput.value += `   Full Path: ${fileData.path}\n`;
        
        if (fileData.bval || fileData.bvec) {
            elements.resultsOutput.value += `   Associated Files:\n`;
            if (fileData.bval) elements.resultsOutput.value += `     - bval: ${fileData.bval}\n`;
            if (fileData.bvec) elements.resultsOutput.value += `     - bvec: ${fileData.bvec}\n`;
        }
        elements.resultsOutput.value += '\n';
    });
    
    elements.resultsOutput.value += '⚙️ EXPORT CONFIGURATION:\n';
    elements.resultsOutput.value += `Selected Atlas: ${elements.processingMethod.value}\n`;
    elements.resultsOutput.value += `DSI Studio Path: ${elements.dsiPathDisplay.textContent}\n\n`;
    
    elements.resultsOutput.value += '✅ READY FOR EXPORT:\n';
    elements.resultsOutput.value += `Files to process: ${selectedExportPaths.length}\n`;
    elements.resultsOutput.value += `Atlas configured: ${elements.processingMethod.value ? 'YES' : 'NO'}\n`;
    elements.resultsOutput.value += `DSI Studio configured: ${!elements.dsiPathDisplay.textContent.includes('No DSI Studio') ? 'YES' : 'NO'}\n\n`;
    
    elements.resultsOutput.value += '🚀 Export process completed with the above configuration...\n';
    elements.resultsOutput.value += 'Ready to generate connectivity matrices from processed dMRI files.\n';
}

// Utility functions
function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function createDragDropHandlers(area, processFunction) {
    return {
        dragover: (e) => {
            e.preventDefault();
            area.classList.add('dragover');
        },
        dragleave: (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
        },
        drop: (e) => {
            e.preventDefault();
            area.classList.remove('dragover');
            processFunction(e.dataTransfer.files);
        }
    };
}

function createFileItem(id, name, path, removeFunction, container) {
    const fileItem = document.createElement('div');
    fileItem.className = 'file-item';
    fileItem.dataset.fileId = id;
    
    fileItem.innerHTML = `
        <button class="file-remove-btn" onclick="${removeFunction}(${id})">×</button>
        <div class="file-item-name">${name}</div>
        <div class="file-item-path">${path}</div>
    `;
    
    // Add click handler for selection
    fileItem.addEventListener('click', (e) => {
        // Don't select if clicking the remove button
        if (e.target.classList.contains('file-remove-btn')) return;
        
        const isFileContainer = container === elements.filesContainer;
        const selectedSet = isFileContainer ? selectedFiles : selectedModels;
        
        // Deselect any previously selected item
        selectedSet.clear();
        const allItems = container.querySelectorAll('.file-item');
        allItems.forEach(item => item.classList.remove('selected'));

        // Select the clicked item
        selectedSet.add(id);
        fileItem.classList.add('selected');
        
        updateSelectionDisplay();
    });
    
    container.appendChild(fileItem);
}

function toggleFileDisplay(clearBtn, hasFiles) {
    if (hasFiles) {
        clearBtn.style.display = 'block';
    } else {
        clearBtn.style.display = 'none';
    }
}

// Function to update DSI Studio path styling
function updateDSIPathStyling() {
    const dsiPathDisplay = elements.dsiPathDisplay;
    const currentText = dsiPathDisplay.textContent.trim();
    const isConfigured = currentText !== 'No DSI Studio path selected - Click me!' && 
                        currentText !== '' && 
                        !currentText.includes('No DSI Studio path selected');
    
    // Remove existing classes
    dsiPathDisplay.classList.remove('unconfigured', 'configured');
    
    // Add appropriate class
    if (isConfigured) {
        dsiPathDisplay.classList.add('configured');
    } else {
        dsiPathDisplay.classList.add('unconfigured');
    }
}

// Backend API class
class BackendAPI {
    constructor(baseUrl = 'http://localhost:3000/api') {
        this.baseUrl = baseUrl;
    }

    async uploadFiles(files, fileType = 'dmri') {
        const formData = new FormData();
        files.forEach(file => {
            formData.append('files', file.file);
        });
        formData.append('fileType', fileType);

        try {
            const response = await fetch(`${this.baseUrl}/upload`, {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Upload failed:', error);
            throw error;
        }
    }

    async runAnalysis(config) {
        try {
            const response = await fetch(`${this.baseUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Analysis failed:', error);
            throw error;
        }
    }

    async getAnalysisStatus(jobId) {
        try {
            const response = await fetch(`${this.baseUrl}/status/${jobId}`);
            if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
            return await response.json();
        } catch (error) {
            console.error('Status check failed:', error);
            throw error;
        }
    }
}

// Initialize API client
const api = new BackendAPI();

// File processing functions - now only store paths locally
function processFiles(files, fileType = 'dmri') {
    let validExtensions;
    
    if (fileType === 'connectivity_matrix') {
        validExtensions = ['.mat'];
    } else {
        validExtensions = ['.nii', '.nii.gz', '.dcm', '.fdf'];
    }
    
    processFileGroup(files, validExtensions, selectedFilePaths, 'file', 
                    elements.filesContainer, 'removeFile', 
                    elements.clearAllBtn, fileType);
}

function processModels(files) {
    const validExtensions = ['.pth'];
    processFileGroup(files, validExtensions, selectedModelPaths, 'model', 
                    elements.filesContainer2, 'removeModel', 
                    elements.clearAllBtn2, 'pytorch_model');
}

function processFileGroup(files, validExtensions, targetArray, type, 
                        container, removeFunction, clearBtn, fileType = 'dmri') {
    const validFiles = [];

    for (let file of files) {
        const fileName = file.name.toLowerCase();
        const isValid = validExtensions.some(ext => fileName.endsWith(ext));
        
        if (isValid && !targetArray.some(item => item.name === file.name)) {
            validFiles.push(file);
        }
    }

    if (validFiles.length === 0) {
        if (files.length > 0) {
            let fileTypeDescription;
            switch(fileType) {
                case 'connectivity_matrix':
                    fileTypeDescription = 'connectivity matrix (.mat)';
                    break;
                case 'pytorch_model':
                    fileTypeDescription = 'PyTorch model (.pth)';
                    break;
                default:
                    fileTypeDescription = 'dMRI';
            }
            alert(`No valid ${fileTypeDescription} files found, or files already added.\nSupported extensions: ${validExtensions.join(', ')}`);
        }
        return;
    }

    // Add files to local storage (just store paths, don't upload)
    validFiles.forEach(file => {
        const fileId = type === 'file' ? ++fileCounter : ++modelCounter;
        
        const fileData = {
            id: fileId,
            name: file.name,
            path: file.path || `Local file: ${file.name}`,
            file: file,
            type: fileType // Add file type information
        };
        
        targetArray.push(fileData);
        createFileItem(fileData.id, fileData.name, fileData.path, removeFunction, container);
    });

    toggleFileDisplay(clearBtn, targetArray.length > 0);
    
    let fileTypeDescription;
    switch(fileType) {
        case 'connectivity_matrix':
            fileTypeDescription = 'connectivity matrix';
            break;
        case 'pytorch_model':
            fileTypeDescription = 'PyTorch model';
            break;
        default:
            fileTypeDescription = 'dMRI';
    }
    console.log(`Added ${validFiles.length} ${fileTypeDescription} files to local storage:`, validFiles.map(f => f.name));
}

function removeFile(fileId) {
    removeFromGroup(fileId, selectedFilePaths, elements.filesContainer,
                    elements.clearAllBtn);
    // Also remove from export paths if it exists there
    const exportIndex = selectedExportPaths.findIndex(file => file.id === fileId);
    if (exportIndex !== -1) {
        selectedExportPaths.splice(exportIndex, 1);
    }
}

function removeModel(fileId) {
    removeFromGroup(fileId, selectedModelPaths, elements.filesContainer2, 
                    elements.clearAllBtn2);
}

function removeFromGroup(fileId, targetArray, container, clearBtn, updateFunction) {
    const index = targetArray.findIndex(file => file.id === fileId);
    if (index !== -1) {
        targetArray.splice(index, 1);
    }
    
    // Remove from selection sets
    selectedFiles.delete(fileId);
    selectedModels.delete(fileId);
    
    const fileItem = container.querySelector(`[data-file-id="${fileId}"]`);
    if (fileItem) {
        fileItem.remove();
    }
    
    updateSelectionDisplay();
    toggleFileDisplay(clearBtn, targetArray.length > 0);
}

function clearAllFiles() {
    clearGroup(selectedFilePaths, elements.filesContainer, elements.fileInput, elements.clearAllBtn);
    // Also clear export paths
    selectedExportPaths.length = 0;
}

function clearAllModels() {
    clearGroup(selectedModelPaths, elements.filesContainer2, elements.fileInput2, elements.clearAllBtn2);
}

function clearGroup(targetArray, container, input, clearBtn) {
    targetArray.length = 0;
    selectedFiles.clear();
    selectedModels.clear();
    container.innerHTML = '';
    input.value = '';
    updateSelectionDisplay();
    toggleFileDisplay(clearBtn, false);
}

function updateSelectionDisplay() {
    const selectedFilesList = Array.from(selectedFiles).map(id => {
        const file = selectedFilePaths.find(f => f.id === id);
        return file ? file.name : null;
    }).filter(Boolean);
    
    const selectedModelsList = Array.from(selectedModels).map(id => {
        const model = selectedModelPaths.find(f => f.id === id);
        return model ? model.name : null;
    }).filter(Boolean);
    
    console.log('Selected files:', selectedFilesList);
    console.log('Selected models:', selectedModelsList);
    
    // Update run button state
    const hasSelectedFiles = selectedFilesList.length > 0;
    const hasSelectedModels = selectedModelsList.length > 0;
    
    if (hasSelectedFiles && hasSelectedModels) {
        elements.runBtn.style.background = 'linear-gradient(45deg, #3b82f6, #1e40af)';
        elements.runBtn.disabled = false;
    } else {
        elements.runBtn.style.background = 'linear-gradient(45deg, #6b7280, #4b5563)';
        elements.runBtn.disabled = false; // Keep enabled to show what's missing
    }
}

// Event Listeners Setup
function initializeEventListeners() {
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        document.addEventListener(eventName, preventDefaults, false);
    });

    // Modal event listeners
    elements.modalCancel.addEventListener('click', () => {
        hideNiftiModal();
        // Reset the process
        pendingNiftiFiles = [];
        currentModalFileIndex = 0;
        selectedExportPaths = [];
        
        elements.resultsOutput.value = '❌ Export process cancelled by user.\n';
    });

    elements.modalSubmit.addEventListener('click', () => {
        const bval = elements.niftiItem1.value.trim();
        const bvec = elements.niftiItem2.value.trim();
        
        if (!bval || !bvec) {
            alert('Please fill in both .bval and .bvec fields before confirming.');
            return;
        }
        
        // Get current file being processed
        const currentFile = pendingNiftiFiles[currentModalFileIndex];
        
        // Find the corresponding file data in selectedExportPaths and add the modal data
        const fileData = selectedExportPaths.find(f => f.name === currentFile.name);
        if (fileData) {
            fileData.bval = bval;
            fileData.bvec = bvec;
        }
        
        hideNiftiModal();
        
        // Move to next file
        currentModalFileIndex++;
        processNextNiftiFile();
    });

    // Close modal when clicking outside
    elements.niftiModal.addEventListener('click', (e) => {
        if (e.target === elements.niftiModal) {
            hideNiftiModal();
            // Reset the process
            pendingNiftiFiles = [];
            currentModalFileIndex = 0;
            selectedExportPaths = [];
            
            elements.resultsOutput.value = '❌ Export process cancelled by user.\n';
        }
    });

    // Handle Enter key in modal inputs
    [elements.niftiItem1, elements.niftiItem2].forEach(input => {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                elements.modalSubmit.click();
            }
        });
    });

    // Section 1 (dMRI) events
    elements.uploadBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        
        // Use a different file selection for CM files (only .mat files)
        try {
            let filePaths;
            
            if (window.electronAPI?.selectFiles) {
                const allFiles = await window.electronAPI.selectFiles();
                
                if (!allFiles || allFiles.length === 0) return;
                
                // Filter only .mat files
                filePaths = allFiles.filter(path => {
                    const fileName = path.toLowerCase();
                    return fileName.endsWith('.mat');
                });
                
                if (filePaths.length === 0) {
                    alert('❌ No .mat files selected. Please select connectivity matrix files (.mat format only).');
                    return;
                }
                
                if (filePaths.length !== allFiles.length) {
                    alert(`⚠️ Only .mat files are allowed for connectivity matrices. Selected ${filePaths.length} valid files out of ${allFiles.length} total files.`);
                }
            } else {
                // Fallback for non-Electron browsers
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.mat';
                input.multiple = true;
                input.style.display = 'none';
                
                const filePromise = new Promise((resolve) => {
                    input.addEventListener('change', (e) => {
                        const files = Array.from(e.target.files);
                        const paths = files.map(file => file.name);
                        resolve(paths);
                    });
                });
                
                document.body.appendChild(input);
                input.click();
                document.body.removeChild(input);
                
                filePaths = await filePromise;
                if (!filePaths || filePaths.length === 0) return;
            }

            // Process the selected .mat files
            filePaths.forEach((path) => {
                const name = path.split(/[\\/]/).pop();
                const fileId = ++fileCounter;

                const fileData = {
                    id: fileId,
                    name,
                    path,
                    type: 'connectivity_matrix' // Mark as CM file
                };

                selectedFilePaths.push(fileData);
                createFileItem(fileId, name, path, 'removeFile', elements.filesContainer);
            });

            toggleFileDisplay(elements.clearAllBtn, selectedFilePaths.length > 0);
            
            // Show success message in results
            elements.resultsOutput.value = '=== CONNECTIVITY MATRIX UPLOAD ===\n\n';
            elements.resultsOutput.value += '📊 UPLOADED CONNECTIVITY MATRICES:\n';
            
            filePaths.forEach((path, index) => {
                const fileName = path.split(/[\\/]/).pop();
                elements.resultsOutput.value += `${index + 1}. ${fileName}\n`;
                elements.resultsOutput.value += `   Path: ${path}\n`;
                elements.resultsOutput.value += `   Type: Connectivity Matrix (.mat)\n\n`;
            });
            
            elements.resultsOutput.value += `✅ Successfully uploaded ${filePaths.length} connectivity matrix file(s).\n`;
            elements.resultsOutput.value += 'Ready for analysis with PyTorch models.\n';
            
        } catch (error) {
            console.error('CM file selection error:', error);
            alert('Error selecting connectivity matrix files. Please try again.');
        }
    });

    elements.uploadAndExportBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        
        // Check if Atlas is selected
        const atlasSelected = elements.processingMethod.value && elements.processingMethod.value.trim() !== '';
        
        // Check if DSI Studio path is configured
        const dsiPathConfigured = elements.dsiPathDisplay.textContent.trim() !== 'No DSI Studio path selected - Click me!' && 
                                 elements.dsiPathDisplay.textContent.trim() !== '' && 
                                 !elements.dsiPathDisplay.textContent.includes('No DSI Studio path selected');
        
        // Show error message if requirements are not met
        if (!atlasSelected || !dsiPathConfigured) {
            let errorMessage = '❌ Cannot export to connectivity matrix. Missing required configuration:\n\n';
            
            if (!atlasSelected) {
                errorMessage += '• Atlas not selected - Please choose an Atlas from the dropdown\n';
            }
            
            if (!dsiPathConfigured) {
                errorMessage += '• DSI Studio path not configured - Please click on the red path box to select DSI Studio executable\n';
            }
            
            errorMessage += '\n⚠️ Both Atlas and DSI Studio path are required for export functionality.';
            
            alert(errorMessage);
            return;
        }
        
        // If all requirements are met, proceed with file selection
        try {
            const filePaths = await window.electronAPI.selectMRIs();
            
            if (!filePaths || filePaths.length === 0) {
                return; // User cancelled file selection
            }
            
            // Reset export state
            selectedExportPaths = [];
            pendingNiftiFiles = [];
            currentModalFileIndex = 0;
            
            // Separate NIfTI files from other files
            const niftiFiles = [];
            const otherFiles = [];
            
            filePaths.forEach((path) => {
                const name = path.split(/[\\/]/).pop();
                const isNifti = name.toLowerCase().endsWith('.nii') || name.toLowerCase().endsWith('.nii.gz');
                
                const fileInfo = { name, path };
                
                if (isNifti) {
                    niftiFiles.push(fileInfo);
                } else {
                    otherFiles.push(fileInfo);
                }
            });
            
            // Process non-NIfTI files immediately
            otherFiles.forEach((fileInfo) => {
                const fileId = ++fileCounter;
                const fileData = {
                    id: fileId,
                    name: fileInfo.name,
                    path: fileInfo.path,
                    type: 'dmri_export'
                };
                selectedExportPaths.push(fileData);
            });
            
            // If there are NIfTI files, start the modal process
            if (niftiFiles.length > 0) {
                // Add NIfTI files to selectedExportPaths first
                niftiFiles.forEach((fileInfo) => {
                    const fileId = ++fileCounter;
                    const fileData = {
                        id: fileId,
                        name: fileInfo.name,
                        path: fileInfo.path,
                        type: 'dmri_export'
                    };
                    selectedExportPaths.push(fileData);
                });
                
                // Set up modal processing
                pendingNiftiFiles = niftiFiles;
                currentModalFileIndex = 0;
                
                // Start showing results
                elements.resultsOutput.value = '=== PROCESSING FILES FOR EXPORT ===\n\n';
                elements.resultsOutput.value += `📁 Found ${niftiFiles.length} NIfTI file(s) requiring additional information.\n`;
                elements.resultsOutput.value += `📁 Found ${otherFiles.length} other file(s) ready for processing.\n\n`;
                elements.resultsOutput.value += '⏳ Please provide .bval and .bvec information for each NIfTI file...\n';
                
                // Start the modal process
                processNextNiftiFile();
            } else {
                // No NIfTI files, process immediately
                finalizeExportProcess();
            }
            
        } catch (error) {
            console.error('File selection error:', error);
            alert('Error selecting files. Please try again.');
        }
    });

    elements.fileInput.addEventListener('change', (e) => {
        // Check if this is being triggered by Upload CM button or Upload and Export button
        // We can determine this by checking if we're in CM upload mode
        const isCMUpload = elements.uploadBtn.textContent.includes('Upload CM');
        const fileType = isCMUpload ? 'connectivity_matrix' : 'dmri';
        processFiles(e.target.files, fileType);
    });
    elements.clearAllBtn.addEventListener('click', clearAllFiles);

    // Section 2 (PyTorch) events
    elements.uploadBtn2.addEventListener('click', async (e) => {
        e.stopPropagation();
        
        // Use a specific file selection for PyTorch models (only .pth files)
        try {
            let filePaths;
            
            if (window.electronAPI?.selectModels) {
                const allFiles = await window.electronAPI.selectModels();
                
                if (!allFiles || allFiles.length === 0) return;
                
                // Filter only .pth files
                filePaths = allFiles.filter(path => {
                    const fileName = path.toLowerCase();
                    return fileName.endsWith('.pth');
                });
                
                if (filePaths.length === 0) {
                    alert('❌ No .pth files selected. Please select PyTorch model files (.pth format only).');
                    return;
                }
                
                if (filePaths.length !== allFiles.length) {
                    alert(`⚠️ Only .pth files are allowed for PyTorch models. Selected ${filePaths.length} valid files out of ${allFiles.length} total files.`);
                }
            } else {
                // Fallback for non-Electron browsers
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = '.pth';
                input.multiple = true;
                input.style.display = 'none';
                
                const filePromise = new Promise((resolve) => {
                    input.addEventListener('change', (e) => {
                        const files = Array.from(e.target.files);
                        const paths = files.map(file => file.name);
                        resolve(paths);
                    });
                });
                
                document.body.appendChild(input);
                input.click();
                document.body.removeChild(input);
                
                filePaths = await filePromise;
                if (!filePaths || filePaths.length === 0) return;
            }

            // Process the selected .pth files
            filePaths.forEach((path) => {
                const name = path.split(/[\\/]/).pop();
                const fileId = ++modelCounter;

                const fileData = {
                    id: fileId,
                    name,
                    path,
                    type: 'pytorch_model' // Mark as PyTorch model file
                };

                selectedModelPaths.push(fileData);
                createFileItem(fileId, name, path, 'removeModel', elements.filesContainer2);
            });

            toggleFileDisplay(elements.clearAllBtn2, selectedModelPaths.length > 0);
            
            // Show success message in results
            elements.resultsOutput.value = '=== PYTORCH MODELS UPLOAD ===\n\n';
            elements.resultsOutput.value += '🔥 UPLOADED PYTORCH MODELS:\n';
            
            filePaths.forEach((path, index) => {
                const fileName = path.split(/[\\/]/).pop();
                elements.resultsOutput.value += `${index + 1}. ${fileName}\n`;
                elements.resultsOutput.value += `   Path: ${path}\n`;
                elements.resultsOutput.value += `   Type: PyTorch Model (.pth)\n\n`;
            });
            
            elements.resultsOutput.value += `✅ Successfully uploaded ${filePaths.length} PyTorch model file(s).\n`;
            elements.resultsOutput.value += 'Ready for analysis with connectivity matrices.\n';
            
        } catch (error) {
            console.error('PyTorch model file selection error:', error);
            alert('Error selecting PyTorch model files. Please try again.');
        }
    });

    elements.fileInput2.addEventListener('change', (e) => processModels(e.target.files));
    elements.clearAllBtn2.addEventListener('click', clearAllModels);

    // DSI path selector
    elements.dsiPathDisplay.addEventListener('click', async () => {
        if (window.electronAPI?.selectDSIStudioPath) {
            const path = await window.electronAPI.selectDSIStudioPath();
            if (path) {
                elements.dsiPathDisplay.textContent = path;
                updateDSIPathStyling(); // Update styling after path selection
            }
        } else {
            // Fallback for non-Electron browsers
            const input = document.createElement('input');
            input.type = 'file';
            input.accept = '.exe';
            input.style.display = 'none';
            input.addEventListener('change', (e) => {
                const file = e.target.files[0];
                if (file) {
                    elements.dsiPathDisplay.textContent = `C:\\Program Files\\DSI Studio\\${file.name}`;
                    updateDSIPathStyling(); // Update styling after path selection
                }
            });
            document.body.appendChild(input);
            input.click();
            document.body.removeChild(input);
        }
    });

    // Run button
    elements.runBtn.addEventListener('click', async () => {
        const selectedFilesList = Array.from(selectedFiles).map(id => {
            const file = selectedFilePaths.find(f => f.id === id);
            return file ? { id: file.id, name: file.name, path: file.path } : null;
        }).filter(Boolean);
        
        const selectedModelsList = Array.from(selectedModels).map(id => {
            const model = selectedModelPaths.find(f => f.id === id);
            return model ? { id: model.id, name: model.name, path: model.path } : null;
        }).filter(Boolean);
        
        // Clear previous results
        elements.resultsOutput.value = '';
        
        // Check if we have selections
        if (selectedFilesList.length === 0 && selectedModelsList.length === 0) {
            elements.resultsOutput.value = 'Please select at least one file or model to run analysis.\n\n';
            elements.resultsOutput.value += 'How to select:\n';
            elements.resultsOutput.value += '• Click on any added dMRI file to select it\n';
            elements.resultsOutput.value += '• Click on any added PyTorch model to select it\n';
            elements.resultsOutput.value += '• Selected items will be highlighted in blue with a checkmark\n';
            return;
        }
        
        elements.resultsOutput.value = '=== ANALYSIS STARTED ===\n\n';
        
        // Display selected files
        if (selectedFilesList.length > 0) {
            elements.resultsOutput.value += '📁 SELECTED dMRI FILES:\n';
            selectedFilesList.forEach((file, index) => {
                elements.resultsOutput.value += `${index + 1}. ${file.name}\n`;
                elements.resultsOutput.value += `   Path: ${file.path || 'N/A'}\n`;
                elements.resultsOutput.value += `   ID: ${file.id}\n\n`;
            });
        } else {
            elements.resultsOutput.value += '⚠️  No dMRI files selected\n\n';
        }
        
        // Display selected models
        if (selectedModelsList.length > 0) {
            elements.resultsOutput.value += '🔥 SELECTED PYTORCH MODELS:\n';
            selectedModelsList.forEach((model, index) => {
                elements.resultsOutput.value += `${index + 1}. ${model.name}\n`;
                elements.resultsOutput.value += `   Path: ${model.path || 'N/A'}\n`;
                elements.resultsOutput.value += `   ID: ${model.id}\n\n`;
            });
        } else {
            elements.resultsOutput.value += '⚠️  No PyTorch models selected\n\n';
        }
        
        // Configuration info
        elements.resultsOutput.value += '⚙️  CONFIGURATION:\n';
        elements.resultsOutput.value += `Atlas: ${elements.processingMethod.value || 'None selected'}\n`;
        elements.resultsOutput.value += `DSI Studio Path: ${elements.dsiPathDisplay.textContent}\n\n`;
        
        // Summary
        elements.resultsOutput.value += '📊 SUMMARY:\n';
        elements.resultsOutput.value += `Total dMRI files selected: ${selectedFilesList.length}\n`;
        elements.resultsOutput.value += `Total PyTorch models selected: ${selectedModelsList.length}\n`;
        elements.resultsOutput.value += `Ready for processing: ${selectedFilesList.length > 0 && selectedModelsList.length > 0 ? 'YES ✅' : 'NO ❌'}\n\n`;
        
        if (selectedFilesList.length > 0 && selectedModelsList.length > 0) {
            elements.resultsOutput.value += '🚀 Starting backend analysis...\n';
            
            try {
                const config = {
                    dmriFiles: selectedFilesList,
                    pytorchModels: selectedModelsList,
                    processingMethod: elements.processingMethod.value,
                    dsiPath: elements.dsiPathDisplay.textContent !== 'No DSI Studio path selected - Click me!' ? 
                            elements.dsiPathDisplay.textContent : null
                };

                elements.resultsOutput.value += 'Submitting job to backend...\n';
                const result = await api.runAnalysis(config);
                
                elements.resultsOutput.value += `✅ Analysis submitted successfully!\n`;
                elements.resultsOutput.value += `Job ID: ${result.jobId || 'N/A'}\n`;
                
            } catch (error) {
                elements.resultsOutput.value += `❌ Backend Error: ${error.message}\n`;
                elements.resultsOutput.value += 'Note: This demo shows selected files even without a real backend.\n';
            }
        } else {
            elements.resultsOutput.value += '❌ Cannot start analysis: Need at least one file AND one model selected.\n';
        }
    });

    // Initialize DSI Studio path styling on page load
    updateDSIPathStyling();
}

// Public API
function getSelectedFilePaths() {
    return selectedFilePaths.map(file => file.path);
}

function getSelectedFiles() {
    return selectedFilePaths;
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    // Set initial styling for DSI Studio path
    updateDSIPathStyling();
});

// Make functions globally accessible
window.getSelectedFilePaths = getSelectedFilePaths;
window.getSelectedFiles = getSelectedFiles;
window.removeFile = removeFile;
window.removeModel = removeModel;