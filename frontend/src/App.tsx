import { useState, useCallback } from 'react';
import { FileUpload, DataPlot, SVDPanel, GAPanel, LDAPanel, DemoSelector } from './components';
import {
  uploadData,
  getSVD,
  runGA,
  runLDA,
  updateIRF,
} from './api/client';
import type {
  SessionState,
  SVDResponse,
  GAResponse,
  LDAResponse,
} from './types';
import './App.css';

type Tab = 'data' | 'svd' | 'ga' | 'lda';

function App() {
  // Session state
  const [session, setSession] = useState<SessionState>({
    sessionId: null,
    filename: null,
    dataInfo: null,
    rawData: null,
  });

  // Analysis results
  const [svdData, setSvdData] = useState<SVDResponse | null>(null);
  const [gaResults, setGaResults] = useState<GAResponse | null>(null);
  const [ldaResults, setLdaResults] = useState<LDAResponse | null>(null);

  // UI state
  const [activeTab, setActiveTab] = useState<Tab>('data');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // IRF state
  const [irfFwhm, setIrfFwhm] = useState(0.1);

  // Handle file upload
  const handleUpload = useCallback(async (file: File) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await uploadData(file);
      setSession({
        sessionId: response.session_id,
        filename: response.filename,
        dataInfo: response.info,
        rawData: response.raw_data,
      });
      // Reset analysis results
      setSvdData(null);
      setGaResults(null);
      setLdaResults(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Handle tab change
  const handleTabChange = useCallback(
    async (tab: Tab) => {
      setActiveTab(tab);
      if (!session.sessionId) return;

      if (tab === 'svd' && !svdData) {
        setIsLoading(true);
        try {
          // Update IRF first
          await updateIRF(session.sessionId, 1, irfFwhm, 0, [0], 500);
          const data = await getSVD(session.sessionId);
          setSvdData(data);
        } catch (err) {
          setError(err instanceof Error ? err.message : 'Failed to load SVD');
        } finally {
          setIsLoading(false);
        }
      }
    },
    [session.sessionId, svdData, irfFwhm]
  );

  // Handle Global Analysis
  const handleRunGA = useCallback(
    async (
      wLSVs: string,
      initialTaus: number[],
      bounds: [number, number][],
      alpha: number,
      fitIRF: boolean
    ) => {
      if (!session.sessionId) return;
      setIsLoading(true);
      setError(null);
      try {
        const results = await runGA(
          session.sessionId,
          wLSVs,
          initialTaus,
          bounds,
          alpha,
          fitIRF
        );
        setGaResults(results);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'GA failed');
      } finally {
        setIsLoading(false);
      }
    },
    [session.sessionId]
  );

  // Handle LDA
  const handleRunLDA = useCallback(
    async (params: Parameters<typeof runLDA>[1]) => {
      if (!session.sessionId) return;
      setIsLoading(true);
      setError(null);
      try {
        const results = await runLDA(session.sessionId, params);
        setLdaResults(results);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'LDA failed');
      } finally {
        setIsLoading(false);
      }
    },
    [session.sessionId]
  );

  return (
    <div className="app">
      <header className="app-header">
        <h1>WebLDM</h1>
        <p>Web-based Lifetime Density Analysis</p>
      </header>

      <main className="app-main">
        {!session.sessionId ? (
          <div className="upload-section">
            <h2>Upload Time-Resolved Spectroscopy Data</h2>
            <p>
              Upload a CSV file with wavelengths in the first row and time points
              in the first column.
            </p>
            <FileUpload onUpload={handleUpload} isLoading={isLoading} />
            
            <DemoSelector
              onDemoLoaded={(response) => {
                setSession({
                  sessionId: response.session_id,
                  filename: response.filename,
                  dataInfo: response.info,
                  rawData: response.raw_data,
                });
                setSvdData(null);
                setGaResults(null);
                setLdaResults(null);
              }}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
              setError={setError}
            />
          </div>
        ) : (
          <>
            <div className="session-info">
              <span>File: {session.filename}</span>
              <span>
                {session.dataInfo?.n_times} time points x{' '}
                {session.dataInfo?.n_wavelengths} wavelengths
              </span>
              <button
                className="new-session"
                onClick={() => {
                  setSession({
                    sessionId: null,
                    filename: null,
                    dataInfo: null,
                    rawData: null,
                  });
                  setSvdData(null);
                  setGaResults(null);
                  setLdaResults(null);
                }}
              >
                New Session
              </button>
            </div>

            <nav className="tabs">
              <button
                className={activeTab === 'data' ? 'active' : ''}
                onClick={() => handleTabChange('data')}
              >
                Data
              </button>
              <button
                className={activeTab === 'svd' ? 'active' : ''}
                onClick={() => handleTabChange('svd')}
              >
                SVD
              </button>
              <button
                className={activeTab === 'ga' ? 'active' : ''}
                onClick={() => handleTabChange('ga')}
              >
                Global Analysis
              </button>
              <button
                className={activeTab === 'lda' ? 'active' : ''}
                onClick={() => handleTabChange('lda')}
              >
                LDA
              </button>
            </nav>

            {error && <div className="error">{error}</div>}

            <div className="tab-content">
              {activeTab === 'data' && session.rawData && (
                <div className="data-tab">
                  <div className="irf-controls">
                    <h4>IRF Parameters</h4>
                    <div className="control-group">
                      <label>FWHM:</label>
                      <input
                        type="number"
                        value={irfFwhm}
                        onChange={(e) => setIrfFwhm(parseFloat(e.target.value))}
                        step={0.01}
                        min={0}
                      />
                    </div>
                  </div>
                  <DataPlot data={session.rawData} />
                </div>
              )}

              {activeTab === 'svd' && (
                <div className="svd-tab">
                  {isLoading ? (
                    <p>Loading SVD...</p>
                  ) : svdData ? (
                    <SVDPanel data={svdData} />
                  ) : (
                    <p>Click to load SVD analysis</p>
                  )}
                </div>
              )}

              {activeTab === 'ga' && session.sessionId && (
                <div className="ga-tab">
                  <GAPanel
                    sessionId={session.sessionId}
                    onRunGA={handleRunGA}
                    gaResults={gaResults}
                    isLoading={isLoading}
                  />
                </div>
              )}

              {activeTab === 'lda' && session.sessionId && (
                <div className="lda-tab">
                  <LDAPanel
                    sessionId={session.sessionId}
                    onRunLDA={handleRunLDA}
                    ldaResults={ldaResults}
                    gaTaus={gaResults?.taus}
                    isLoading={isLoading}
                  />
                </div>
              )}
            </div>
          </>
        )}
      </main>

      <footer className="app-footer">
        <p>
          Based on{' '}
          <a
            href="https://github.com/gadorlhiac/PyLDM"
            target="_blank"
            rel="noopener noreferrer"
          >
            PyLDM
          </a>{' '}
          by Dorlhiac GF, Fare C, van Thor JJ.
        </p>
      </footer>
    </div>
  );
}

export default App;
