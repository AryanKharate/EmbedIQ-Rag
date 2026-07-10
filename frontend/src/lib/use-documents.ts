import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { documentsApi, type ApiDocument } from "./api";
import { toast } from "sonner";

export type { ApiDocument };

/* ─── Query keys ─── */
const DOCS_KEY = ["documents"] as const;

/* ─── Hooks ─── */

export function useDocuments() {
  return useQuery({
    queryKey: DOCS_KEY,
    queryFn: documentsApi.list,
    staleTime: 30_000,
  });
}

export function useUploadDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: documentsApi.upload,
    onSuccess: (doc) => {
      qc.setQueryData<ApiDocument[]>(DOCS_KEY, (old = []) => [doc, ...old]);
      toast.success(`Uploaded "${doc.filename}"`);
    },
    onError: (err: Error) => {
      toast.error(`Upload failed: ${err.message}`);
    },
  });
}

export function useToggleDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, is_active }: { id: string; is_active: boolean }) =>
      documentsApi.toggle(id, is_active),
    onMutate: async ({ id, is_active }) => {
      await qc.cancelQueries({ queryKey: DOCS_KEY });
      const prev = qc.getQueryData<ApiDocument[]>(DOCS_KEY);
      qc.setQueryData<ApiDocument[]>(DOCS_KEY, (old = []) =>
        old.map((d) => (d.id === id ? { ...d, is_active } : d)),
      );
      return { prev };
    },
    onError: (_err, _vars, ctx) => {
      qc.setQueryData(DOCS_KEY, ctx?.prev);
      toast.error("Failed to update document status");
    },
    onSettled: () => qc.invalidateQueries({ queryKey: DOCS_KEY }),
  });
}

export function useDeleteDocument() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: documentsApi.delete,
    onMutate: async (id) => {
      await qc.cancelQueries({ queryKey: DOCS_KEY });
      const prev = qc.getQueryData<ApiDocument[]>(DOCS_KEY);
      qc.setQueryData<ApiDocument[]>(DOCS_KEY, (old = []) =>
        old.filter((d) => d.id !== id),
      );
      return { prev };
    },
    onError: (_err, _vars, ctx) => {
      qc.setQueryData(DOCS_KEY, ctx?.prev);
      toast.error("Failed to delete document");
    },
    onSuccess: () => toast.success("Document deleted"),
    onSettled: () => qc.invalidateQueries({ queryKey: DOCS_KEY }),
  });
}
