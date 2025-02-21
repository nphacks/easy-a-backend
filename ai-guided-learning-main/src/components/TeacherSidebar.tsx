
import { Link, useLocation } from "react-router-dom";
import { BookOpen, FileText, Users, FileSpreadsheet, Upload, List, PlusSquare } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

const sidebarLinks = [
  {
    title: "Dashboard",
    icon: FileSpreadsheet,
    path: "/teacher-dashboard"
  },
  {
    title: "Upload Notes",
    icon: Upload,
    path: "/teacher-dashboard/upload-notes"
  },
  {
    title: "All Notes",
    icon: BookOpen,
    path: "/teacher-dashboard/notes"
  },
  {
    title: "Generate Assignment",
    icon: PlusSquare,
    path: "/teacher-dashboard/generate-assignment"
  },
  {
    title: "Assignments",
    icon: FileText,
    path: "/teacher-dashboard/assignments"
  },
  {
    title: "Students",
    icon: Users,
    path: "/teacher-dashboard/students"
  },
  {
    title: "Reports",
    icon: List,
    path: "/teacher-dashboard/reports"
  }
];

export const TeacherSidebar = () => {
  const location = useLocation();

  return (
    <div className="w-64 bg-white border-r h-screen fixed left-0 top-0 overflow-y-auto">
      <div className="p-6">
        <Link to="/teacher-dashboard" className="text-xl font-bold text-primary block mb-6">
          Teacher Portal
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
