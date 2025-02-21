
import { useState } from "react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Menu, X } from "lucide-react";

export const Navbar = () => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <nav className="bg-white border-b sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex items-center">
            <Link to="/" className="text-xl font-bold text-primary">
              AI-Guided Learning
            </Link>
          </div>

          {/* Desktop Menu */}
          <div className="hidden md:flex items-center space-x-8">
            <Link to="/" className="text-gray-600 hover:text-secondary transition-colors">
              Home
            </Link>
            <Link to="/features" className="text-gray-600 hover:text-secondary transition-colors">
              Features
            </Link>
            <Link to="/about" className="text-gray-600 hover:text-secondary transition-colors">
              About
            </Link>
            <div className="flex items-center space-x-4">
              <Link to="/login">
                <Button variant="outline" className="border-secondary text-secondary hover:bg-secondary hover:text-white">
                  Login
                </Button>
              </Link>
              <Link to="/signup">
                <Button className="bg-secondary text-white hover:bg-secondary/90">
                  Sign Up
                </Button>
              </Link>
            </div>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-gray-600 hover:text-secondary focus:outline-none"
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="md:hidden py-4">
            <div className="flex flex-col space-y-4">
              <Link
                to="/"
                className="text-gray-600 hover:text-secondary transition-colors"
                onClick={() => setIsOpen(false)}
              >
                Home
              </Link>
              <Link
                to="/features"
                className="text-gray-600 hover:text-secondary transition-colors"
                onClick={() => setIsOpen(false)}
              >
                Features
              </Link>
              <Link
                to="/about"
                className="text-gray-600 hover:text-secondary transition-colors"
                onClick={() => setIsOpen(false)}
              >
                About
              </Link>
              <Link to="/login" onClick={() => setIsOpen(false)}>
                <Button variant="outline" className="w-full border-secondary text-secondary hover:bg-secondary hover:text-white">
                  Login
                </Button>
              </Link>
              <Link to="/signup" onClick={() => setIsOpen(false)}>
                <Button className="w-full bg-secondary text-white hover:bg-secondary/90">
                  Sign Up
                </Button>
              </Link>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};
