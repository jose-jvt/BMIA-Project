// src/App.js
import React from 'react';
import './App.css';
import ImageProcessor from './components/ImageProcessor';

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Procesador de Imágenes Médicas</h1>
      </header>
      <main>
        <ImageProcessor />
      </main>
    </div>
  );
}

export default App;
