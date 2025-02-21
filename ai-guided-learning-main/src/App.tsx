
import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Navbar } from "./components/Navbar";
import { Footer } from "./components/Footer";
import Index from "./pages/Index";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import NotFound from "./pages/NotFound";
import TeacherDashboard from "./pages/TeacherDashboard";
import StudentDashboard from "./pages/StudentDashboard";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <div className="flex flex-col min-h-screen">
          <Routes>
            <Route
              path="/"
              element={
                <>
                  <Navbar />
                  <main className="flex-grow">
                    <Index />
                  </main>
                  <Footer />
                </>
              }
            />
            <Route
              path="/login"
              element={
                <>
                  <Navbar />
                  <main className="flex-grow">
                    <Login />
                  </main>
                  <Footer />
                </>
              }
            />
            <Route
              path="/signup"
              element={
                <>
                  <Navbar />
                  <main className="flex-grow">
                    <Signup />
                  </main>
                  <Footer />
                </>
              }
            />
            <Route path="/teacher-dashboard/*" element={<TeacherDashboard />} />
            <Route path="/student-dashboard" element={<StudentDashboard />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </div>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
