import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Search, ShoppingBag, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

export default function Navbar() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const navigate = useNavigate();

  function handleSearch(e: React.FormEvent) {
    e.preventDefault();
    if (searchQuery.trim()) {
      navigate(`/products?search=${encodeURIComponent(searchQuery.trim())}`);
      setSearchQuery("");
      setMobileOpen(false);
    }
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b bg-white shadow-sm">
      <div className="container mx-auto px-4">
        <div className="flex h-16 items-center justify-between gap-4">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2 font-bold text-xl text-gray-900 shrink-0">
            <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
              <span className="text-white text-xs font-bold">RS</span>
            </div>
            RunShop
          </Link>

          {/* Desktop nav */}
          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-gray-600">
            <Link to="/" className="hover:text-gray-900 transition-colors">Trang Chủ</Link>
            <Link to="/products" className="hover:text-gray-900 transition-colors">Sản Phẩm</Link>
            <Link to="/products?categories=Trail+Running" className="hover:text-gray-900 transition-colors">Trail</Link>
            <Link to="/products?categories=Road+Running" className="hover:text-gray-900 transition-colors">Road</Link>
          </nav>

          {/* Search bar */}
          <form onSubmit={handleSearch} className="hidden md:flex flex-1 max-w-sm">
            <div className="relative w-full">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
              <Input
                type="search"
                placeholder="Tìm kiếm giày..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 bg-gray-50 border-gray-200"
              />
            </div>
          </form>

          {/* Right actions */}
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="icon" className="hidden md:flex">
              <ShoppingBag className="h-5 w-5" />
            </Button>
            <Button variant="ghost" size="icon" className="md:hidden" onClick={() => setMobileOpen(!mobileOpen)}>
              {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </Button>
          </div>
        </div>

        {/* Mobile menu */}
        {mobileOpen && (
          <div className="md:hidden border-t py-4 flex flex-col gap-4">
            <form onSubmit={handleSearch}>
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
                <Input
                  type="search"
                  placeholder="Tìm kiếm giày..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-9"
                />
              </div>
            </form>
            <nav className="flex flex-col gap-2 text-sm font-medium">
              <Link to="/" className="py-2 px-2 rounded hover:bg-gray-50" onClick={() => setMobileOpen(false)}>Trang Chủ</Link>
              <Link to="/products" className="py-2 px-2 rounded hover:bg-gray-50" onClick={() => setMobileOpen(false)}>Tất Cả Sản Phẩm</Link>
              <Link to="/products?categories=Trail+Running" className="py-2 px-2 rounded hover:bg-gray-50" onClick={() => setMobileOpen(false)}>Trail Running</Link>
              <Link to="/products?categories=Road+Running" className="py-2 px-2 rounded hover:bg-gray-50" onClick={() => setMobileOpen(false)}>Road Running</Link>
            </nav>
          </div>
        )}
      </div>
    </header>
  );
}
