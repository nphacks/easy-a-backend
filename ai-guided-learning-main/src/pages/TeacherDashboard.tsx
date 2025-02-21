import { Routes, Route } from "react-router-dom";
import { motion } from "framer-motion";
import { TeacherSidebar } from "@/components/TeacherSidebar";
import { BookOpen, Users, Calendar, BarChart } from "lucide-react";
import { Card } from "@/components/ui/card";

// Dashboard Overview Component
const DashboardOverview = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="text-3xl font-bold text-primary mb-8">Dashboard Overview</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-purple-100 rounded-full">
              <Users className="h-6 w-6 text-purple-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Total Students</p>
              <h3 className="text-2xl font-bold text-primary">156</h3>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-blue-100 rounded-full">
              <BookOpen className="h-6 w-6 text-blue-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Active Courses</p>
              <h3 className="text-2xl font-bold text-primary">8</h3>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-green-100 rounded-full">
              <Calendar className="h-6 w-6 text-green-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Upcoming Sessions</p>
              <h3 className="text-2xl font-bold text-primary">12</h3>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-orange-100 rounded-full">
              <BarChart className="h-6 w-6 text-orange-600" />
            </div>
            <div>
              <p className="text-sm text-gray-600">Average Performance</p>
              <h3 className="text-2xl font-bold text-primary">85%</h3>
            </div>
          </div>
        </Card>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center justify-between py-2 border-b">
                <div>
                  <p className="font-medium">Assignment Submitted</p>
                  <p className="text-sm text-gray-600">Student #{i} submitted Math Assignment</p>
                </div>
                <span className="text-sm text-gray-500">2h ago</span>
              </div>
            ))}
          </div>
        </Card>

        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <h2 className="text-xl font-semibold mb-4">Upcoming Classes</h2>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="flex items-center justify-between py-2 border-b">
                <div>
                  <p className="font-medium">Advanced Mathematics</p>
                  <p className="text-sm text-gray-600">Class {i} • 25 Students</p>
                </div>
                <span className="text-sm text-gray-500">9:00 AM</span>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </motion.div>
  );
};

// Upload Notes Component
const UploadNotes = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="text-3xl font-bold text-primary mb-8">Upload Notes</h1>
      <Card className="p-6 bg-white/80 backdrop-blur-sm">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Title
            </label>
            <input
              type="text"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
              placeholder="Enter note title"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description
            </label>
            <textarea
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
              rows={4}
              placeholder="Enter note description"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Upload File
            </label>
            <input
              type="file"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
              accept=".pdf,.doc,.docx"
            />
          </div>
          <button className="bg-secondary text-white px-4 py-2 rounded-lg hover:bg-secondary/90">
            Upload Note
          </button>
        </div>
      </Card>
    </motion.div>
  );
};

// Notes List Component
const NotesList = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="text-3xl font-bold text-primary mb-8">All Notes</h1>
      <Card className="p-6 bg-white/80 backdrop-blur-sm">
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="flex items-center justify-between p-4 border rounded-lg">
              <div>
                <h3 className="font-medium">Mathematics Note #{i}</h3>
                <p className="text-sm text-gray-600">Uploaded 2 days ago</p>
              </div>
              <button className="text-secondary hover:text-secondary/80">
                Generate Assignment
              </button>
            </div>
          ))}
        </div>
      </Card>
    </motion.div>
  );
};

// Generate Assignment Component
const GenerateAssignment = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="text-3xl font-bold text-primary mb-8">Generate Assignment</h1>
      <Card className="p-6 bg-white/80 backdrop-blur-sm">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Select Note
            </label>
            <select className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent">
              <option>Mathematics Note #1</option>
              <option>Mathematics Note #2</option>
              <option>Mathematics Note #3</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Number of Questions
            </label>
            <input
              type="number"
              min="1"
              max="20"
              className="w-full px-4 py-2 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
              placeholder="Enter number of questions"
            />
          </div>
          <button className="bg-secondary text-white px-4 py-2 rounded-lg hover:bg-secondary/90">
            Generate Assignment
          </button>
        </div>
      </Card>
    </motion.div>
  );
};

// Main TeacherDashboard Component
const TeacherDashboard = () => {
  return (
    <div className="min-h-screen bg-accent">
      <TeacherSidebar />
      <div className="ml-64 p-6">
        <div className="max-w-7xl mx-auto">
          <Routes>
            <Route index element={<DashboardOverview />} />
            <Route path="upload-notes" element={<UploadNotes />} />
            <Route path="notes" element={<NotesList />} />
            <Route path="generate-assignment" element={<GenerateAssignment />} />
            {/* Additional routes will be added for other sections */}
          </Routes>
        </div>
      </div>
    </div>
  );
};

export default TeacherDashboard;
