import { useState, useCallback } from "react";

type AsyncFn<T, A extends unknown[]> = (...args: A) => Promise<T>;

export function useAsync<T, A extends unknown[]>(fn: AsyncFn<T, A>) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const execute = useCallback(
    async (...args: A) => {
      setLoading(true);
      setError(null);
      try {
        const result = await fn(...args);
        setData(result);
        return result;
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        setError(msg);
        return null;
      } finally {
        setLoading(false);
      }
    },
    [fn],
  );

  return { data, error, loading, execute };
}
