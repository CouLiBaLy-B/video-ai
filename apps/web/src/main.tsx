import React, { FormEvent, useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

type JobStatus =
  | 'queued'
  | 'analyzing'
  | 'planning'
  | 'waiting_for_approval'
  | 'generating'
  | 'reviewing'
  | 'completed'
  | 'failed'
  | 'cancelled';

type PlanStepResponse = {
  name: string;
  description: string;
  agent: string;
};

type JobEventResponse = {
  status: JobStatus;
  message: string;
  created_at: string;
};

type JobResponse = {
  id: string;
  status: JobStatus;
  status_reason: string | null;
  prompt: string;
  created_at: string;
  updated_at: string;
  video_url: string | null;
  plan_steps: PlanStepResponse[];
  events: JobEventResponse[];
};

type ComponentHealth = {
  status: string;
  detail: string | null;
};

type SystemHealth = {
  api: ComponentHealth;
  storage: ComponentHealth;
  job_repository: ComponentHealth;
  vllm_text: ComponentHealth;
  vllm_vision: ComponentHealth;
};

type SystemCapabilities = {
  agent_planner_provider: string;
  ai_provider: string;
  default_video_backend: string;
  available_video_backends: string[];
  planned_video_backends: string[];
  text_model: string;
  vision_model: string;
};

type TimelineItem = {
  label: string;
  status: 'pending' | 'active' | 'done' | 'failed';
};

const statusLabels: Record<JobStatus, string> = {
  queued: 'Requête reçue',
  analyzing: 'Analyse de l’image',
  planning: 'Planification agentique',
  waiting_for_approval: 'Validation humaine requise',
  generating: 'Génération vidéo',
  reviewing: 'Contrôle qualité',
  completed: 'Vidéo prête',
  failed: 'Échec',
  cancelled: 'Annulé'
};

function App() {
  const [prompt, setPrompt] = useState('Transforme cette image en vidéo cinématique avec un lent mouvement de caméra.');
  const [image, setImage] = useState<File | null>(null);
  const [job, setJob] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [capabilities, setCapabilities] = useState<SystemCapabilities | null>(null);
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [backend, setBackend] = useState('mock');
  const [width, setWidth] = useState(512);
  const [height, setHeight] = useState(512);
  const [numFrames, setNumFrames] = useState(121);
  const [fps, setFps] = useState(24);
  const [seed, setSeed] = useState('');
  const [guidanceScale, setGuidanceScale] = useState(3.5);
  const [inferenceSteps, setInferenceSteps] = useState(30);

  useEffect(() => {
    fetch('/api/system/capabilities')
      .then((response) => response.ok ? response.json() : null)
      .then((payload: SystemCapabilities | null) => {
        if (!payload) return;
        setCapabilities(payload);
        setBackend(payload.default_video_backend);
      })
      .catch(() => undefined);
    fetch('/api/system/health')
      .then((response) => response.ok ? response.json() : null)
      .then((payload: SystemHealth | null) => setHealth(payload))
      .catch(() => undefined);
  }, []);

  const timeline = useMemo<TimelineItem[]>(() => buildTimeline(job?.status), [job?.status]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (!image) {
      setError('Ajoute une image avant de générer la vidéo.');
      return;
    }
    setIsSubmitting(true);
    try {
      const form = new FormData();
      form.append('prompt', prompt);
      form.append('image', image);
      form.append('requested_backend', backend);
      form.append('width', String(width));
      form.append('height', String(height));
      form.append('num_frames', String(numFrames));
      form.append('fps', String(fps));
      form.append('guidance_scale', String(guidanceScale));
      form.append('inference_steps', String(inferenceSteps));
      if (seed.trim()) form.append('seed', seed.trim());
      const response = await fetch('/api/generations', { method: 'POST', body: form });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const created = (await response.json()) as JobResponse;
      setJob(created);
      await pollJob(created.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Erreur inconnue');
    } finally {
      setIsSubmitting(false);
    }
  }

  async function pollJob(jobId: string) {
    for (let index = 0; index < 120; index += 1) {
      const response = await fetch(`/api/generations/${jobId}`);
      if (!response.ok) return;
      const nextJob = (await response.json()) as JobResponse;
      setJob(nextJob);
      if (['waiting_for_approval', 'completed', 'failed', 'cancelled'].includes(nextJob.status)) return;
      await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }


  async function approveGeneration() {
    if (!job) return;
    setError(null);
    setIsSubmitting(true);
    try {
      const response = await fetch(`/api/generations/${job.id}/approve`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      const approved = (await response.json()) as JobResponse;
      setJob(approved);
      await pollJob(approved.id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Erreur inconnue');
    } finally {
      setIsSubmitting(false);
    }
  }

  async function rejectGeneration() {
    if (!job) return;
    setError(null);
    try {
      const response = await fetch(`/api/generations/${job.id}/reject`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      setJob((await response.json()) as JobResponse);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Erreur inconnue');
    }
  }


  async function cancelGeneration() {
    if (!job) return;
    setError(null);
    try {
      const response = await fetch(`/api/generations/${job.id}/cancel`, { method: 'POST' });
      if (!response.ok) throw new Error(await response.text());
      setJob((await response.json()) as JobResponse);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : 'Erreur inconnue');
    }
  }

  return (
    <main className="app-shell">
      <section className="sidebar">
        <div className="brand-mark">VA</div>
        <h1>Video AI</h1>
        <p>Interface agentique text + image → vidéo avec DeepAgents, vLLM et modèles open source.</p>
      </section>

      <section className="chat-panel">
        <div className="message assistant">
          <strong>Assistant vidéo</strong>
          <p>Envoie une image et un prompt. Je vais analyser, planifier, générer puis vérifier la vidéo.</p>
        </div>

        <form className="composer" onSubmit={submit}>
          <textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} rows={4} />
          <label className="upload-box">
            <input
              type="file"
              accept="image/png,image/jpeg,image/webp"
              onChange={(event) => setImage(event.target.files?.[0] ?? null)}
            />
            <span>{image ? image.name : 'Glisse ou sélectionne une image'}</span>
          </label>
          <div className="controls-grid">
            <label>
              Backend
              <select value={backend} onChange={(event) => setBackend(event.target.value)}>
                {(capabilities?.available_video_backends ?? ['mock', 'ltx-video']).map((item) => (
                  <option key={item} value={item}>{item}</option>
                ))}
              </select>
            </label>
            <label>
              Largeur
              <input type="number" min="64" max="4096" value={width} onChange={(event) => setWidth(Number(event.target.value))} />
            </label>
            <label>
              Hauteur
              <input type="number" min="64" max="4096" value={height} onChange={(event) => setHeight(Number(event.target.value))} />
            </label>
            <label>
              Frames
              <input type="number" min="1" max="1000" value={numFrames} onChange={(event) => setNumFrames(Number(event.target.value))} />
            </label>
            <label>
              FPS
              <input type="number" min="1" max="120" value={fps} onChange={(event) => setFps(Number(event.target.value))} />
            </label>
            <label>
              Seed
              <input value={seed} placeholder="auto" onChange={(event) => setSeed(event.target.value)} />
            </label>
            <label>
              Guidance
              <input type="number" min="0" max="30" step="0.1" value={guidanceScale} onChange={(event) => setGuidanceScale(Number(event.target.value))} />
            </label>
            <label>
              Steps
              <input type="number" min="1" max="200" value={inferenceSteps} onChange={(event) => setInferenceSteps(Number(event.target.value))} />
            </label>
          </div>
          {capabilities && (
            <p className="capabilities-line">
              Planner: {capabilities.agent_planner_provider} · AI: {capabilities.ai_provider} · VLM: {capabilities.vision_model}
            </p>
          )}
          {health && (
            <div className="health-line">
              {Object.entries(health).map(([name, component]) => (
                <span key={name} className={component.status === 'ok' ? 'ok' : 'warn'}>
                  {name}: {component.status}
                </span>
              ))}
            </div>
          )}

          <button disabled={isSubmitting}>{isSubmitting ? 'Génération...' : 'Générer la vidéo'}</button>
        </form>

        {error && <div className="error">{error}</div>}

        {job && (
          <div className="message assistant result-card">
            <strong>{statusLabels[job.status]}</strong>
            {job.status_reason && <p>{job.status_reason}</p>}
            <ol className="timeline">
              {timeline.map((item) => (
                <li key={item.label} className={item.status}>{item.label}</li>
              ))}
            </ol>
            {job.plan_steps.length > 0 && (
              <div className="plan-card">
                <h3>Plan agentique</h3>
                {job.plan_steps.map((step) => (
                  <div key={step.name} className="plan-step">
                    <span>{step.agent}</span>
                    <strong>{step.name}</strong>
                    <p>{step.description}</p>
                  </div>
                ))}
              </div>
            )}
            {['queued', 'planning', 'analyzing', 'generating', 'reviewing'].includes(job.status) && (
              <div className="cancel-card">
                <button type="button" className="secondary" onClick={cancelGeneration}>
                  Annuler la génération
                </button>
              </div>
            )}
            {job.status === 'waiting_for_approval' && (
              <div className="approval-card">
                <strong>Validation GPU requise</strong>
                <p>Cette génération utilise un backend coûteux. Confirme avant de lancer le worker GPU.</p>
                <div>
                  <button type="button" onClick={approveGeneration} disabled={isSubmitting}>
                    Valider la génération GPU
                  </button>
                  <button type="button" className="secondary" onClick={rejectGeneration}>
                    Rejeter
                  </button>
                </div>
              </div>
            )}
            {job.events.length > 0 && (
              <div className="event-card">
                <h3>Journal d’exécution</h3>
                {job.events.map((event) => (
                  <p key={`${event.status}-${event.created_at}`}>
                    <span>{statusLabels[event.status]}</span> — {event.message}
                  </p>
                ))}
              </div>
            )}
            {job.video_url && (
              <div className="video-card">
                <video src={job.video_url} controls />
                <a href={job.video_url} download>
                  Télécharger la vidéo
                </a>
              </div>
            )}
          </div>
        )}
      </section>
    </main>
  );
}

function buildTimeline(status?: JobStatus): TimelineItem[] {
  const order: JobStatus[] = ['queued', 'analyzing', 'planning', 'generating', 'reviewing', 'completed'];
  const currentIndex = status ? order.indexOf(status) : -1;
  return order.map((item, index) => ({
    label: statusLabels[item],
    status:
      status === 'failed'
        ? 'failed'
        : index < currentIndex
          ? 'done'
          : index === currentIndex
            ? 'active'
            : 'pending'
  }));
}

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
