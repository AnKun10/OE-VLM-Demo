import { useContext } from "react";
import { ToastContext } from "../components/Toaster";

export function useToast() {
  return useContext(ToastContext);
}
