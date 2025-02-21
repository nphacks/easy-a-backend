
import { Link } from "react-router-dom";
import { Facebook, Twitter, Instagram, Youtube } from "lucide-react";

export const Footer = () => {
  return (
    <footer className="bg-white border-t">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">AI-Guided Learning</h3>
            <p className="text-gray-600 text-sm">
              Redefining education through AI-powered guidance and personalized learning experiences.
            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li><Link to="/" className="text-gray-600 hover:text-secondary transition-colors">Home</Link></li>
              <li><Link to="/about" className="text-gray-600 hover:text-secondary transition-colors">About</Link></li>
              <li><Link to="/features" className="text-gray-600 hover:text-secondary transition-colors">Features</Link></li>
              <li><Link to="/contact" className="text-gray-600 hover:text-secondary transition-colors">Contact</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Resources</h3>
            <ul className="space-y-2">
              <li><Link to="/blog" className="text-gray-600 hover:text-secondary transition-colors">Blog</Link></li>
              <li><Link to="/help" className="text-gray-600 hover:text-secondary transition-colors">Help Center</Link></li>
              <li><Link to="/terms" className="text-gray-600 hover:text-secondary transition-colors">Terms of Service</Link></li>
              <li><Link to="/privacy" className="text-gray-600 hover:text-secondary transition-colors">Privacy Policy</Link></li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Connect With Us</h3>
            <div className="flex space-x-4">
              <a href="#" className="text-gray-600 hover:text-secondary transition-colors">
                <Facebook className="h-5 w-5" />
              </a>
              <a href="#" className="text-gray-600 hover:text-secondary transition-colors">
                <Twitter className="h-5 w-5" />
              </a>
              <a href="#" className="text-gray-600 hover:text-secondary transition-colors">
                <Instagram className="h-5 w-5" />
              </a>
              <a href="#" className="text-gray-600 hover:text-secondary transition-colors">
                <Youtube className="h-5 w-5" />
              </a>
            </div>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t text-center text-gray-600">
          <p>&copy; {new Date().getFullYear()} AI-Guided Learning. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};
