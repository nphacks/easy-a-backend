
import { Link, useLocation } from "react-router-dom";
import { BookOpen, FileText, Home, MessageSquare, Award } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const sidebarLinks = [
  {
    title: "Dashboard",
    icon: Home,
    path: "/student-dashboard"
  },
  {
    title: "Course Materials",
    icon: BookOpen,
    path: "/student-dashboard/materials"
  },
  {
    title: "Assignments",
    icon: FileText,
    path: "/student-dashboard/assignments"
  },
  {
    title: "Study Help",
    icon: MessageSquare,
    path: "/student-dashboard/study-help"
  },
  {
    title: "Progress",
    icon: Award,
    path: "/student-dashboard/progress"
  }
];

export const StudentSidebar = () => {
  const location = useLocation();

  return (
    <div className="w-64 bg-white border-r h-screen fixed left-0 top-0 overflow-y-auto">
      <div className="p-6">
        <Link to="/student-dashboard" className="text-xl font-bold text-primary block mb-6">
          Student Portal
        </Link>
        <nav className="space-y-2">
          {sidebarLinks.map((link) => {
            const Icon = link.icon;
            const isActive = location.pathname === link.path;
            
            return (
              <Link key={link.path} to={link.path}>
                <Button
                  variant="ghost"
                  className={cn(
                    "w-full justify-start",
                    isActive && "bg-secondary/10 text-secondary hover:bg-secondary/20"
                  )}
                >
                  <Icon className="mr-2 h-5 w-5" />
                  {link.title}
                </Button>
              </Link>
            );
          })}
        </nav>
      </div>
    </div>
  );
};
