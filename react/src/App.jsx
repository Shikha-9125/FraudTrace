import React from 'react';
import FileUpload from './components/FileUpload'; // adjust path if needed
import './index.css'; // this must be here

function App() {
  return (
    <div className="App">
      {/* <h1 className="text-2xl font-bold text-center my-6">Fraud Detection Upload</h1> */}
      
      <FileUpload />
    </div>
  );
}

export default App;
