import { useCallback, useState } from "react";
import { uploadFile } from "../lib/uploadFile";
import type { AttachmentRef } from "../types";

export function useFileUpload(): {
  uploading: boolean;
  upload: (f: File) => Promise<AttachmentRef>;
} {
  const [uploading, setUploading] = useState(false);
  const upload = useCallback(async (f: File) => {
    setUploading(true);
    try {
      return await uploadFile(f);
    } finally {
      setUploading(false);
    }
  }, []);
  return { uploading, upload };
}
