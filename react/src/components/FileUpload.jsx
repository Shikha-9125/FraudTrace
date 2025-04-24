

import React, { useState } from 'react';
import axios from 'axios';
import {
  PieChart, Pie, Cell, Tooltip as PieTooltip, Legend as PieLegend, ResponsiveContainer as PieResponsiveContainer
} from 'recharts';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as BarTooltip, Legend as BarLegend, ResponsiveContainer as BarResponsiveContainer
} from 'recharts';

const FileUpload = () => {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const PIEC_COLORS = ['#0088FE', '#FF8042'];

  const handleFileChange = e => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.endsWith('.csv')) {
      setFile(selectedFile);
    }
  };

  const handleUpload = async () => {
    if (!file) return alert('Please select a file first.');
    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    try {
      const { data } = await axios.post('http://localhost:5000/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: progressEvent => {
          const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(percent);
        }
      });
      setResult(data);
    } catch (err) {
      console.error('Upload failed:', err);
      alert('Upload failed.');
    } finally {
      setLoading(false);
      setUploadProgress(0);
    }
  };

  // New: download handler
  const handleDownload = async () => {
    try {
      const res = await axios.get('http://localhost:5000/download-report', {
        responseType: 'blob'
      });
      const blob = new Blob([res.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', 'fraud_report.pdf');
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Download failed:', err);
      alert('Could not download report.');
    }
  };

  const fraudData = result
    ? Object.entries(result.isFraud_counts).map(([cls, count]) => ({ name: cls === '0' ? 'No Fraud' : 'Fraud', value: count }))
    : [];

  const confusionData = result
    ? [
        { name: 'True Negative', value: result.confusion_matrix[0][0] },
        { name: 'False Positive', value: result.confusion_matrix[0][1] },
        { name: 'False Negative', value: result.confusion_matrix[1][0] },
        { name: 'True Positive', value: result.confusion_matrix[1][1] }
      ]
    : [];

  return (
    <div className="relative min-h-screen flex flex-col items-center justify-center text-white">
      {/* Background */}
      <div
        className="absolute inset-0 bg-cover bg-center"
        style={{ backgroundImage: `url("https://t3.ftcdn.net/jpg/08/16/90/22/360_F_816902258_Fq09muOHUVmLhQkzpn8EkL5ItZptEiRQ.jpg")`, opacity: 0.9 }}
      />

      {/* Title */}
      <h1 className="relative z-10 text-6xl font-extrabold tracking-tight text-white drop-shadow-2xl mb-30 text-center">
        Fraud Detection System
      </h1>

      {/* Upload Card */}
      <div className="relative z-10 bg-white/30 backdrop-blur-lg p-10 rounded-2xl shadow-2xl max-w-4xl w-full text-white">

        <div className="text-center mb-10">
          <p className="text-xl font-medium text-white/90 drop-shadow-md">
            Upload your financial data to detect potential fraud using <span className="text-indigo-200 font-semibold">AI</span>
          </p>
        </div>

        {/* Dropzone */}
        <div
          className="relative w-full p-6 border-2 border-dashed border-white/60 rounded-lg hover:border-white transition-colors cursor-pointer mb-6 text-center"
          onDrop={e => {
            e.preventDefault();
            const dropped = e.dataTransfer.files[0];
            if (dropped && dropped.name.endsWith('.csv')) setFile(dropped);
          }}
          onDragOver={e => e.preventDefault()}
        >
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="absolute inset-0 opacity-0 cursor-pointer"
          />
          <p className="text-sm text-white/70">Drag & drop CSV or click to browse</p>
        </div>

        {/* File name preview */}
        {file && (
          <p className="text-center text-white/80 mb-4">
            Selected file: <span className="font-semibold">{file.name}</span>
          </p>
        )}

        {/* Upload button */}
        <div className="flex justify-center mb-4">
          <button
            onClick={handleUpload}
            className="bg-indigo-600 hover:bg-indigo-800 transition text-white px-6 py-2 rounded-lg shadow-lg font-semibold w-full sm:w-auto"
          >
            {loading ? 'Processing...' : 'Upload'}
          </button>
        </div>

        {/* Progress bar */}
        {loading && (
          <div className="w-full bg-gray-300 rounded-full h-2 mt-2 overflow-hidden">
            <div className="bg-green-500 h-full transition-all duration-300" style={{ width: `${uploadProgress}%` }} />
          </div>
        )}

        {/* Results section */}
        {result && (
          <div className="space-y-12 mt-10">
            <div>
              <h2 className="text-3xl font-semibold mb-4 text-white text-center">Fraud Distribution</h2>
              <PieResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie data={fraudData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={80} label>
                    {fraudData.map((entry, idx) => (
                      <Cell key={idx} fill={PIEC_COLORS[idx % PIEC_COLORS.length]} />
                    ))}
                  </Pie>
                  <PieTooltip />
                  <PieLegend />
                </PieChart>
              </PieResponsiveContainer>
            </div>

            <div>
              <h2 className="text-2xl font-semibold mb-4 text-white">Confusion Matrix</h2>
              <BarResponsiveContainer width="100%" height={300}>
                <BarChart data={confusionData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" stroke="#fff" />
                  <YAxis stroke="#fff" allowDecimals={false} />
                  <BarTooltip />
                  <BarLegend />
                  <Bar dataKey="value" fill="#82ca9d" />
                </BarChart>
              </BarResponsiveContainer>
            </div>

            <div className="flex justify-center">
              <button
                onClick={handleDownload}
                className="bg-green-600 hover:bg-green-700 transition text-white px-6 py-2 rounded-lg shadow font-semibold"
              >
                Download Report
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FileUpload;

