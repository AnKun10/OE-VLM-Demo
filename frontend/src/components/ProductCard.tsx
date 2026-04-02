import { Link } from "react-router-dom";
import { Heart, MessageCircle } from "lucide-react";
import type { Product } from "@/types";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";

interface ProductCardProps {
  product: Product;
}

export default function ProductCard({ product }: ProductCardProps) {
  const [imgError, setImgError] = useState(false);
  const [wishlisted, setWishlisted] = useState(false);

  const imageSrc = !imgError && product.image_url
    ? product.image_url
    : "https://placehold.co/600x600?text=No+Image";

  return (
    <Link to={`/products/${product.id}`} className="group block">
      <div className="bg-white rounded-xl overflow-hidden shadow-[0_4px_20px_rgba(0,0,0,0.06)] hover:shadow-[0_8px_30px_rgba(0,0,0,0.12)] transition-all duration-200">
        {/* Image */}
        <div className="relative aspect-square overflow-hidden bg-white">
          <img
            src={imageSrc}
            alt={product.name}
            onError={() => setImgError(true)}
            className="w-full h-full object-contain object-center group-hover:scale-105 transition-transform duration-300"
          />

          {/* Hover overlay */}
          <div className="absolute inset-0 bg-black/5 opacity-0 group-hover:opacity-100 transition-opacity duration-200" />

          {/* Hover action buttons */}
          <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex gap-2 opacity-0 group-hover:opacity-100 translate-y-2 group-hover:translate-y-0 transition-all duration-200">
            <button
              onClick={(e) => { e.preventDefault(); setWishlisted(!wishlisted); }}
              title="Yêu thích"
              className="w-10 h-10 rounded-full bg-white shadow-md flex items-center justify-center hover:bg-gray-50 transition-colors"
            >
              <Heart className={`h-4 w-4 transition-colors ${wishlisted ? "fill-red-500 text-red-500" : "text-gray-500"}`} />
            </button>
            <button
              onClick={(e) => {
                e.preventDefault();
                window.dispatchEvent(new CustomEvent("add-to-chat", {
                  detail: { imageUrl: imageSrc, productName: product.name },
                }));
              }}
              title="Hỏi trợ lý về sản phẩm này"
              className="w-10 h-10 rounded-full bg-white shadow-md flex items-center justify-center hover:bg-gray-50 transition-colors"
            >
              <MessageCircle className="h-4 w-4 text-[#015e9f]" />
            </button>
          </div>
        </div>

        {/* Info */}
        <div className="p-3 pt-[13px] space-y-2.5">
          <div className="flex items-center justify-between gap-3">
            <h3 className="min-w-0 text-[15px] font-semibold text-[#444956] line-clamp-2 leading-[1.42] tracking-[-0.01em] group-hover:text-[#0d1b67] transition-colors">
              {product.name}
            </h3>
            <span className="shrink-0 max-w-[42%] truncate text-[11px] font-semibold uppercase tracking-[0.08em] text-[#8b90a0]">
              #{product.id.slice(-5)}
            </span>
          </div>

          <div className="flex items-center justify-between gap-2">
            <Badge variant="outline" className="max-w-[48%] truncate border-[#0d1b67]/15 bg-[#f4f7ff] text-[#0d1b67]">
              {product.store}
            </Badge>
            <Badge variant="outline" className="max-w-[48%] truncate border-[#2f6f55]/15 bg-[#edf8f2] text-[#2f6f55]">
              {product.layer}
            </Badge>
          </div>
        </div>
      </div>
    </Link>
  );
}
