
import { useState } from "react";
import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";

const Signup = () => {
  const [role, setRole] = useState<"student" | "teacher" | null>(null);

  return (
    <div className="min-h-screen bg-accent py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="bg-white rounded-2xl shadow-sm p-8"
        >
          <div className="text-center mb-8">
            <h2 className="text-3xl font-bold text-primary mb-2">Create Account</h2>
            <p className="text-gray-600">Join our learning community today</p>
          </div>

          {!role ? (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-center mb-4">I want to join as...</h3>
              <Button
                onClick={() => setRole("student")}
                className="w-full bg-secondary text-white hover:bg-secondary/90 mb-3"
              >
                Student
              </Button>
              <Button
                onClick={() => setRole("teacher")}
                className="w-full bg-secondary text-white hover:bg-secondary/90"
              >
                Teacher
              </Button>
            </div>
          ) : (
            <form className="space-y-4">
              <div>
                <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">
                  Full Name
                </label>
                <input
                  id="name"
                  type="text"
                  className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
                  placeholder="Enter your full name"
                />
              </div>
              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                  Email Address
                </label>
                <input
                  id="email"
                  type="email"
                  className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
                  placeholder="Enter your email"
                />
              </div>
              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
                  Password
                </label>
                <input
                  id="password"
                  type="password"
                  className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
                  placeholder="Choose a password"
                />
              </div>
              <Button type="submit" className="w-full bg-secondary text-white hover:bg-secondary/90">
                Sign Up as {role === "student" ? "Student" : "Teacher"}
              </Button>
              <button
                type="button"
                onClick={() => setRole(null)}
                className="w-full text-sm text-gray-600 hover:text-secondary"
              >
                Change Role
              </button>
            </form>
          )}

          <div className="mt-6 text-center text-sm">
            <span className="text-gray-600">Already have an account?</span>{" "}
            <Link to="/login" className="text-secondary hover:text-secondary/90 font-medium">
              Login
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Signup;
