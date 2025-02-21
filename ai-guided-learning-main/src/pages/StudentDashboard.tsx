import { useState } from "react";
import { Routes, Route } from "react-router-dom";
import { motion } from "framer-motion";
import { StudentSidebar } from "@/components/StudentSidebar";
import { BookOpen, GraduationCap, Clock, Trophy, MessageSquare, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

const DashboardOverview = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-primary mb-8">Student Dashboard</h1>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="p-6 bg-white/80 backdrop-blur-sm">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-blue-100 rounded-full">
                <BookOpen className="h-6 w-6 text-blue-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Enrolled Courses</p>
                <h3 className="text-2xl font-bold text-primary">5</h3>
              </div>
            </div>
          </Card>

          <Card className="p-6 bg-white/80 backdrop-blur-sm">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-green-100 rounded-full">
                <GraduationCap className="h-6 w-6 text-green-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Completed Courses</p>
                <h3 className="text-2xl font-bold text-primary">3</h3>
              </div>
            </div>
          </Card>

          <Card className="p-6 bg-white/80 backdrop-blur-sm">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-purple-100 rounded-full">
                <Clock className="h-6 w-6 text-purple-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Study Hours</p>
                <h3 className="text-2xl font-bold text-primary">42h</h3>
              </div>
            </div>
          </Card>

          <Card className="p-6 bg-white/80 backdrop-blur-sm">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-yellow-100 rounded-full">
                <Trophy className="h-6 w-6 text-yellow-600" />
              </div>
              <div>
                <p className="text-sm text-gray-600">Achievements</p>
                <h3 className="text-2xl font-bold text-primary">8</h3>
              </div>
            </div>
          </Card>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card className="p-6 bg-white/80 backdrop-blur-sm">
            <h2 className="text-xl font-semibold mb-4">Current Progress</h2>
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center justify-between py-2 border-b">
                  <div>
                    <p className="font-medium">Mathematics 101</p>
                    <p className="text-sm text-gray-600">Chapter {i} Progress</p>
                  </div>
                  <span className="text-sm font-medium text-green-600">85%</span>
                </div>
              ))}
            </div>
          </Card>

          <Card className="p-6 bg-white/80 backdrop-blur-sm">
            <h2 className="text-xl font-semibold mb-4">Upcoming Assignments</h2>
            <div className="space-y-4">
              {[1, 2, 3].map((i) => (
                <div key={i} className="flex items-center justify-between py-2 border-b">
                  <div>
                    <p className="font-medium">Physics Assignment</p>
                    <p className="text-sm text-gray-600">Due in {i} days</p>
                  </div>
                  <span className="text-sm text-orange-500">Pending</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>
    </motion.div>
  );
};

const MaterialsView = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="text-3xl font-bold text-primary mb-8">Course Materials</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {[
          { title: "Physics", lessons: 12 },
          { title: "Mathematics", lessons: 15 },
          { title: "Chemistry", lessons: 10 },
          { title: "Biology", lessons: 8 },
        ].map((course, idx) => (
          <Card key={idx} className="p-6 bg-white/80 backdrop-blur-sm">
            <div className="flex flex-col space-y-4">
              <div className="p-4 bg-primary/5 rounded-lg">
                <BookOpen className="h-8 w-8 text-primary" />
              </div>
              <h3 className="text-lg font-semibold">{course.title}</h3>
              <p className="text-sm text-gray-600">{course.lessons} Lessons</p>
              <div className="flex space-x-2">
                <Button variant="outline" className="flex-1">View Notes</Button>
                <Button className="flex-1">Start Learning</Button>
              </div>
            </div>
          </Card>
        ))}
      </div>
    </motion.div>
  );
};

const StudyHelpView = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="text-3xl font-bold text-primary mb-8">Study Help</h1>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <h2 className="text-xl font-semibold mb-4">Ask a Question</h2>
          <div className="space-y-4">
            <textarea
              className="w-full p-4 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
              rows={4}
              placeholder="Type your question here..."
            />
            <div className="flex justify-end">
              <Button>Submit Question</Button>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <h2 className="text-xl font-semibold mb-4">Recent Questions</h2>
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <div key={i} className="p-4 border rounded-lg">
                <p className="font-medium mb-2">How do I solve quadratic equations?</p>
                <div className="flex items-center justify-between text-sm text-gray-600">
                  <span>Mathematics</span>
                  <span>2 answers</span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </motion.div>
  );
};

const ProgressView = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <h1 className="text-3xl font-bold text-primary mb-8">Your Progress</h1>
      <div className="space-y-6">
        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <h2 className="text-xl font-semibold mb-4">Overall Performance</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="p-4 border rounded-lg text-center">
              <p className="text-sm text-gray-600">Average Score</p>
              <p className="text-2xl font-bold text-primary mt-1">85%</p>
            </div>
            <div className="p-4 border rounded-lg text-center">
              <p className="text-sm text-gray-600">Assignments Completed</p>
              <p className="text-2xl font-bold text-primary mt-1">24/30</p>
            </div>
            <div className="p-4 border rounded-lg text-center">
              <p className="text-sm text-gray-600">Study Streak</p>
              <p className="text-2xl font-bold text-primary mt-1">7 days</p>
            </div>
          </div>
        </Card>

        <Card className="p-6 bg-white/80 backdrop-blur-sm">
          <h2 className="text-xl font-semibold mb-4">Subject Performance</h2>
          <div className="space-y-4">
            {[
              { subject: "Mathematics", score: 90 },
              { subject: "Physics", score: 85 },
              { subject: "Chemistry", score: 78 },
              { subject: "Biology", score: 88 },
            ].map((subject, idx) => (
              <div key={idx} className="space-y-2">
                <div className="flex justify-between">
                  <span className="font-medium">{subject.subject}</span>
                  <span className="text-gray-600">{subject.score}%</span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-primary rounded-full transition-all duration-500"
                    style={{ width: `${subject.score}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </Card>
      </div>
    </motion.div>
  );
};

const ChatbotPanel = ({ isOpen, onClose, title }: { isOpen: boolean; onClose: () => void; title: string }) => {
  const [messages, setMessages] = useState<Array<{ text: string; isBot: boolean }>>([
    { text: "How can I help you today?", isBot: true }
  ]);
  const [input, setInput] = useState("");

  const handleSend = () => {
    if (input.trim()) {
      setMessages(prev => [...prev, { text: input, isBot: false }]);
      setInput("");
      setTimeout(() => {
        setMessages(prev => [...prev, { text: "Let me help you with that...", isBot: true }]);
      }, 1000);
    }
  };

  return (
    <div className={cn(
      "fixed right-0 top-0 h-screen w-80 bg-white border-l transform transition-transform duration-300",
      isOpen ? "translate-x-0" : "translate-x-full"
    )}>
      <div className="p-4 border-b flex justify-between items-center">
        <h3 className="font-semibold">{title}</h3>
        <Button variant="ghost" size="icon" onClick={onClose}>
          <X className="h-4 w-4" />
        </Button>
      </div>
      <div className="flex flex-col h-[calc(100vh-64px)]">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg, idx) => (
            <div key={idx} className={cn(
              "p-3 rounded-lg max-w-[80%]",
              msg.isBot ? "bg-secondary/10 mr-auto" : "bg-primary text-white ml-auto"
            )}>
              {msg.text}
            </div>
          ))}
        </div>
        <div className="p-4 border-t">
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSend()}
              className="flex-1 px-4 py-2 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
              placeholder="Type your message..."
            />
            <Button onClick={handleSend}>Send</Button>
          </div>
        </div>
      </div>
    </div>
  );
};

const AssignmentView = () => {
  const [chatbotOpen, setChatbotOpen] = useState(false);
  const [chatbotTitle, setChatbotTitle] = useState("");

  const openChatbot = (title: string) => {
    setChatbotTitle(title);
    setChatbotOpen(true);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="relative"
    >
      <h1 className="text-3xl font-bold text-primary mb-8">Current Assignment</h1>
      <Card className="p-6 bg-white/80 backdrop-blur-sm">
        <div className="space-y-8">
          {[1, 2, 3].map((i) => (
            <div key={i} className="space-y-4">
              <div className="flex items-start justify-between">
                <div 
                  className="flex-1 select-none" 
                  onCopy={(e) => e.preventDefault()}
                >
                  <h3 className="font-semibold mb-2">Question {i}</h3>
                  <p className="text-gray-600">What is the significance of Newton's Third Law of Motion?</p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => openChatbot(`Clarify Question ${i}`)}
                  className="ml-4"
                >
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Clarify Question
                </Button>
              </div>
              <div className="flex items-start space-x-4">
                <textarea
                  className="flex-1 p-4 border rounded-lg focus:ring-2 focus:ring-secondary focus:border-transparent"
                  rows={4}
                  placeholder="Write your answer here..."
                  onPaste={(e) => e.preventDefault()}
                />
                <Button
                  variant="outline"
                  onClick={() => openChatbot("Get Help with Answer")}
                >
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Get Help
                </Button>
              </div>
            </div>
          ))}
          <div className="flex justify-end space-x-4 mt-6">
            <Button variant="outline">Save Draft</Button>
            <Button>Submit Assignment</Button>
          </div>
        </div>
      </Card>
      
      <ChatbotPanel 
        isOpen={chatbotOpen}
        onClose={() => setChatbotOpen(false)}
        title={chatbotTitle}
      />
    </motion.div>
  );
};

const StudentDashboard = () => {
  return (
    <div className="min-h-screen bg-accent">
      <StudentSidebar />
      <div className="ml-64 p-6">
        <div className="max-w-7xl mx-auto">
          <Routes>
            <Route index element={<DashboardOverview />} />
            <Route path="assignments" element={<AssignmentView />} />
            <Route path="materials" element={<MaterialsView />} />
            <Route path="study-help" element={<StudyHelpView />} />
            <Route path="progress" element={<ProgressView />} />
          </Routes>
        </div>
      </div>
    </div>
  );
};

export default StudentDashboard;
