import { useState, useEffect } from 'react';
import { getDemoFiles, loadDemoFile, type DemoFile } from '../api/client';

interface DemoSelectorProps {
  onDemoLoaded: (response: {
    session_id: string;
    filename: string;
    info: any;
    raw_data: any;
  }) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export function DemoSelector({
  onDemoLoaded,
  isLoading,
  setIsLoading,
  setError,
}: DemoSelectorProps) {
  const [demos, setDemos] = useState<DemoFile[]>([]);
  const [loadingDemos, setLoadingDemos] = useState(true);

  useEffect(() => {
    async function fetchDemos() {
      try {
        const response = await getDemoFiles();
        setDemos(response.demos);
      } catch (err) {
        console.error('Failed to load demo list:', err);
      } finally {
        setLoadingDemos(false);
      }
    }
    fetchDemos();
  }, []);

  const handleLoadDemo = async (demoId: string) => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await loadDemoFile(demoId);
      onDemoLoaded({
        session_id: response.session_id,
        filename: response.filename,
        info: response.info,
        raw_data: response.raw_data,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load demo');
    } finally {
      setIsLoading(false);
    }
  };

  if (loadingDemos) {
    return <div className="demo-selector loading">Loading demo files...</div>;
  }

  if (demos.length === 0) {
    return null;
  }

  return (
    <div className="demo-selector">
      <h3>Or try a demo dataset</h3>
      <div className="demo-grid">
        {demos.map((demo) => (
          <button
            key={demo.id}
            className="demo-card"
            onClick={() => handleLoadDemo(demo.id)}
            disabled={isLoading}
          >
            <span className="demo-name">{demo.name}</span>
            <span className="demo-description">{demo.description}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
