import React, { FormEvent, useMemo, useState } from 'react';
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
      if (['completed', 'failed', 'cancelled'].includes(nextJob.status)) return;
      await new Promise((resolve) => setTimeout(resolve, 1000));
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
