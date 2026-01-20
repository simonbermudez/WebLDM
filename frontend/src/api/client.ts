import axios from 'axios';
import type {
  UploadResponse,
  SVDResponse,
  GAResponse,
  LDAResponse,
  TSVDResponse,
  ChirpFitResponse,
  RawDataPlot,
  DataInfo,
} from '../types';

// Use relative URL in production (Docker), absolute in development
const API_BASE = import.meta.env.PROD ? '/api' : 'http://localhost:8000/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Upload data file
export async function uploadData(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await api.post<UploadResponse>('/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

// Get data info
export async function getDataInfo(sessionId: string): Promise<{
  info: DataInfo;
  raw_data: RawDataPlot;
}> {
  const response = await api.get(`/data/${sessionId}`);
  return response.data;
}

// Update data bounds
export async function updateBounds(
  sessionId: string,
  wlLb: number,
  wlUb: number,
  t0: number,
  t: number
): Promise<{ info: DataInfo; raw_data: RawDataPlot }> {
  const response = await api.post('/bounds', {
    session_id: sessionId,
    wl_lb: wlLb,
    wl_ub: wlUb,
    t0,
    t,
  });
  return response.data;
}

// Update IRF parameters
export async function updateIRF(
  sessionId: string,
  order: number,
  fwhm: number,
  munot: number,
  mus: number[],
  lamnot: number
): Promise<{ status: string }> {
  const response = await api.post('/irf', {
    session_id: sessionId,
    order,
    fwhm,
    munot,
    mus,
    lamnot,
  });
  return response.data;
}

// Fit chirp
export async function fitChirp(sessionId: string): Promise<ChirpFitResponse> {
  const response = await api.post<ChirpFitResponse>('/chirp/fit', {
    session_id: sessionId,
  });
  return response.data;
}

// Get SVD results
export async function getSVD(sessionId: string): Promise<SVDResponse> {
  const response = await api.get<SVDResponse>(`/svd/${sessionId}`);
  return response.data;
}

// Run Global Analysis
export async function runGA(
  sessionId: string,
  wLSVs: string,
  initialTaus: number[],
  bounds: [number, number][],
  alpha: number = 0,
  fitIRF: boolean = false
): Promise<GAResponse> {
  const response = await api.post<GAResponse>('/ga', {
    session_id: sessionId,
    wLSVs,
    initial_taus: initialTaus,
    bounds,
    alpha,
    fit_irf: fitIRF,
  });
  return response.data;
}

// Run LDA
export async function runLDA(
  sessionId: string,
  params: {
    tauMin?: number;
    tauMax?: number;
    nTaus?: number;
    alphaMin?: number;
    alphaMax?: number;
    nAlphas?: number;
    regMethod?: string;
    regMatrix?: string;
    simfit?: boolean;
    gaTaus?: number[];
  }
): Promise<LDAResponse> {
  const response = await api.post<LDAResponse>('/lda', {
    session_id: sessionId,
    tau_min: params.tauMin ?? -1,
    tau_max: params.tauMax ?? 4,
    n_taus: params.nTaus ?? 100,
    alpha_min: params.alphaMin ?? -4,
    alpha_max: params.alphaMax ?? 2,
    n_alphas: params.nAlphas ?? 20,
    reg_method: params.regMethod ?? 'L2',
    reg_matrix: params.regMatrix ?? 'Id',
    simfit: params.simfit ?? true,
    ga_taus: params.gaTaus,
  });
  return response.data;
}

// Run TSVD
export async function runTSVD(
  sessionId: string,
  k: number,
  tauMin: number = -1,
  tauMax: number = 4,
  nTaus: number = 100,
  gaTaus?: number[]
): Promise<TSVDResponse> {
  const response = await api.post<TSVDResponse>('/tsvd', {
    session_id: sessionId,
    k,
    tau_min: tauMin,
    tau_max: tauMax,
    n_taus: nTaus,
    ga_taus: gaTaus,
  });
  return response.data;
}

// Delete session
export async function deleteSession(sessionId: string): Promise<void> {
  await api.delete(`/session/${sessionId}`);
}

// Demo data types
export interface DemoFile {
  id: string;
  filename: string;
  name: string;
  description: string;
}

export interface DemoListResponse {
  demos: DemoFile[];
}

export interface DemoLoadResponse extends UploadResponse {
  name: string;
}

// Get list of demo files
export async function getDemoFiles(): Promise<DemoListResponse> {
  const response = await api.get<DemoListResponse>('/demos');
  return response.data;
}

// Load a demo file
export async function loadDemoFile(demoId: string): Promise<DemoLoadResponse> {
  const response = await api.post<DemoLoadResponse>(`/demos/${demoId}/load`);
  return response.data;
}

export default api;
