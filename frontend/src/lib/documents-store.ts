export interface RagDocument {
  id: string;
  name: string;
  sizeLabel: string;
  active: boolean;
}

const DOCS_KEY = "rag.documents.v1";

const SEED: RagDocument[] = [
  { id: "doc-1", name: "Product Handbook.pdf", sizeLabel: "1.2 MB", active: true },
  { id: "doc-2", name: "Q3 Financials.xlsx", sizeLabel: "384 KB", active: true },
  { id: "doc-3", name: "Onboarding Guide.docx", sizeLabel: "612 KB", active: false },
  { id: "doc-4", name: "API Reference.md", sizeLabel: "88 KB", active: true },
  { id: "doc-5", name: "Legal Policy.pdf", sizeLabel: "2.4 MB", active: false },
];

export function loadDocuments(): RagDocument[] {
  if (typeof window === "undefined") return SEED;
  try {
    const raw = window.localStorage.getItem(DOCS_KEY);
    if (!raw) return SEED;
    const parsed = JSON.parse(raw) as RagDocument[];
    return Array.isArray(parsed) && parsed.length > 0 ? parsed : SEED;
  } catch {
    return SEED;
  }
}

export function saveDocuments(docs: RagDocument[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(DOCS_KEY, JSON.stringify(docs));
}
