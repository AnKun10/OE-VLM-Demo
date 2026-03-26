import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer className="border-t bg-gray-900 text-gray-300">
      <div className="container mx-auto px-4 py-10">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <div className="w-8 h-8 bg-blue-500 rounded-full flex items-center justify-center">
                <span className="text-white text-xs font-bold">RS</span>
              </div>
              <span className="font-bold text-white text-lg">RunShop</span>
            </div>
            <p className="text-sm leading-relaxed">
              Cửa hàng giày chạy bộ chính hãng hàng đầu Việt Nam. Chất lượng đảm bảo, giao hàng toàn quốc.
            </p>
          </div>
          <div>
            <h3 className="font-semibold text-white mb-3">Thương Hiệu</h3>
            <ul className="space-y-2 text-sm">
              {["HOKA", "Nike", "Adidas", "Puma", "Brooks", "ASICS", "New Balance"].map((b) => (
                <li key={b}>
                  <Link to={`/products?brands=${b}`} className="hover:text-white transition-colors">{b}</Link>
                </li>
              ))}
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-white mb-3">Danh Mục</h3>
            <ul className="space-y-2 text-sm">
              <li><Link to="/products?categories=Road+Running" className="hover:text-white transition-colors">Road Running</Link></li>
              <li><Link to="/products?categories=Trail+Running" className="hover:text-white transition-colors">Trail Running</Link></li>
              <li><Link to="/products?sort_by=discount" className="hover:text-white transition-colors">Khuyến Mãi</Link></li>
              <li><Link to="/products?is_new=true" className="hover:text-white transition-colors">Hàng Mới</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="font-semibold text-white mb-3">Hỗ Trợ</h3>
            <ul className="space-y-2 text-sm">
              <li className="hover:text-white cursor-pointer transition-colors">Hướng dẫn chọn size</li>
              <li className="hover:text-white cursor-pointer transition-colors">Chính sách đổi trả</li>
              <li className="hover:text-white cursor-pointer transition-colors">Giao hàng</li>
              <li className="hover:text-white cursor-pointer transition-colors">Liên hệ</li>
            </ul>
          </div>
        </div>
        <div className="border-t border-gray-700 mt-8 pt-6 text-center text-sm">
          <p>© 2024 RunShop. Tất cả quyền được bảo lưu.</p>
        </div>
      </div>
    </footer>
  );
}
