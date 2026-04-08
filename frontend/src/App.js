// src/App.js
import { useState } from "react";
import axios from "axios";
import "./App.css";

export default function App() {
  const [company, setCompany] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    if (!company || !file) {
      setError("Please enter company name and upload a PDF.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("company", company);
      formData.append("file", file);

      const response = await axios.post(
        "http://127.0.0.1:8000/analyze",
        formData,
        { headers: { "Content-Type": "multipart/form-data" } }
      );

      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.error || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const getSignalStyle = (signal) => {
    if (signal === "Buy")  return "signal buy";
    if (signal === "Sell") return "signal sell";
    return "signal hold";
  };

  const getSentimentStyle = (overall) => {
    if (overall === "positive") return "badge positive";
    if (overall === "negative") return "badge negative";
    return "badge neutral";
  };

  return (
    <div className="app">

      {/* Header */}
      <div className="header">
        <h1>Earnings Call Analyst</h1>
        <p>Upload an earnings call transcript and get instant AI-powered analysis</p>
      </div>

      {/* Input Card */}
      <div className="card">
        <div className="input-row">
          <div className="input-group">
            <label>Company Name</label>
            <input
              type="text"
              placeholder="e.g. HDFC Bank, BOI, TCS"
              value={company}
              onChange={(e) => setCompany(e.target.value)}
            />
          </div>

          <div className="input-group">
            <label>Earnings Call PDF</label>
            <input
              type="file"
              accept=".pdf"
              onChange={(e) => setFile(e.target.files[0])}
            />
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        <button
          className="btn"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? "Analyzing..." : "Analyze"}
        </button>
      </div>

      {/* Loading */}
      {loading && (
        <div className="card center">
          <div className="spinner"></div>
          <p>Running pipeline...</p>
          <p className="muted">This may take 20-30 seconds</p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="results">

          {/* Top row — Signal + Sentiment */}
          <div className="top-row">
            <div className="card center">
              <p className="label">Investment Signal</p>
              <div className={getSignalStyle(result.report.signal)}>
                {result.report.signal}
              </div>
              <p className="muted">{result.report.signal_reason}</p>
            </div>

            <div className="card center">
              <p className="label">Overall Sentiment</p>
              <div className={getSentimentStyle(result.sentiment.overall)}>
                {result.sentiment.overall}
              </div>
              <div className="scores">
                <span className="pos">+{result.sentiment.scores.positive}%</span>
                <span className="neu">~{result.sentiment.scores.neutral}%</span>
                <span className="neg">-{result.sentiment.scores.negative}%</span>
              </div>
            </div>
          </div>

          {/* Summary Sections */}
          <div className="card">
            <h2>Summary</h2>

            {result.report.summary.macroeconomic_environment && (
              <div className="summary-section">
                <h3>Macroeconomic Environment</h3>
                <p>{result.report.summary.macroeconomic_environment}</p>
              </div>
            )}
            {result.report.summary.business_growth && (
              <div className="summary-section">
                <h3>Business Growth & Performance</h3>
                <p>{result.report.summary.business_growth}</p>
              </div>
            )}
            {result.report.summary.digital_initiatives && (
              <div className="summary-section">
                <h3>Digital & Product Initiatives</h3>
                <p>{result.report.summary.digital_initiatives}</p>
              </div>
            )}
            {result.report.summary.profitability && (
              <div className="summary-section">
                <h3>Profitability & Financial Metrics</h3>
                <p>{result.report.summary.profitability}</p>
              </div>
            )}
            {result.report.summary.deposits_advances_casa && (
              <div className="summary-section">
                <h3>Deposits, Advances & CASA</h3>
                <p>{result.report.summary.deposits_advances_casa}</p>
              </div>
            )}
            {result.report.summary.guidance_outlook && (
              <div className="summary-section">
                <h3>Guidance & Outlook</h3>
                <p>{result.report.summary.guidance_outlook}</p>
              </div>
            )}
          </div>

          {/* Key Highlights */}
          <div className="card">
            <h2>Key Highlights</h2>
            <ul>
              {result.report.key_highlights.map((h, i) => (
                <li key={i}>{h}</li>
              ))}
            </ul>
          </div>

          {/* Risks */}
          <div className="card">
            <h2>Risk Analysis</h2>
            {Object.entries(result.report.risks).map(([key, value]) => (
              <div className="summary-section risk" key={key}>
                <h3>{key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</h3>
                <p>{value}</p>
              </div>
            ))}
          </div>

        </div>
      )}

    </div>
  );
}