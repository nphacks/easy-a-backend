import { motion } from "framer-motion";
import { RocketIcon, GraduationCapIcon, BrainCircuitIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

const FeatureCard = ({ icon: Icon, title, description }: { icon: any; title: string; description: string }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.5 }}
    className="relative p-6 rounded-xl bg-white/80 backdrop-blur-sm border border-gray-100 shadow-sm hover:shadow-md transition-all duration-300"
  >
    <div className="flex items-start gap-4">
      <div className="p-2 rounded-lg bg-secondary/10">
        <Icon className="w-6 h-6 text-secondary" />
      </div>
      <div>
        <h3 className="text-lg font-semibold text-primary mb-2">{title}</h3>
        <p className="text-gray-600">{description}</p>
      </div>
    </div>
  </motion.div>
);

const Index = () => {
  return (
    <div className="min-h-screen bg-accent">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12 md:py-20">
        {/* Hero Section */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16 md:mb-24"
        >
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2, duration: 0.5 }}
            className="inline-block px-4 py-1.5 rounded-full bg-secondary/10 text-secondary text-sm font-medium mb-6"
          >
            Redefining AI in Education
          </motion.div>
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.5 }}
            className="text-4xl md:text-5xl lg:text-6xl font-bold text-primary mb-6 leading-tight"
          >
            The Rise of AI-Guided Learning
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.5 }}
            className="text-lg md:text-xl text-gray-600 max-w-3xl mx-auto mb-8"
          >
            Instead of replacing student effort, our AI-powered system guides students through assignments step by step,
            helping them think critically, understand concepts, and develop their own solutions.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5, duration: 0.5 }}
          >
            <Button
              size="lg"
              className="bg-secondary text-white hover:bg-secondary/90 transition-colors duration-300"
            >
              Get Started
            </Button>
          </motion.div>
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 mb-16">
          <FeatureCard
            icon={BrainCircuitIcon}
            title="Interactive Learning"
            description="Get AI-driven guidance, hints, and brainstorming support tailored to your learning style."
          />
          <FeatureCard
            icon={RocketIcon}
            title="Personalized Assistance"
            description="Learn at your own pace with structured AI coaching that adapts to your progress."
          />
          <FeatureCard
            icon={GraduationCapIcon}
            title="Teacher Assurance"
            description="Track real learning progress and ensure concept mastery with detailed analytics."
          />
        </div>

        {/* Why Choose Us Section */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="py-16 bg-white"
        >
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold text-primary mb-4">
                Why Choose AI-Guided Learning?
              </h2>
              <p className="text-gray-600 max-w-2xl mx-auto">
                Our platform combines the power of AI with proven educational methodologies to create
                a unique learning experience that adapts to your needs.
              </p>
            </div>
            <div className="grid md:grid-cols-2 gap-8">
              {/* Add content for Why Choose Us section */}
            </div>
          </div>
        </motion.section>

        {/* Testimonials Section */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="py-16 bg-secondary/5"
        >
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center mb-12">
              <h2 className="text-3xl md:text-4xl font-bold text-primary mb-4">
                What Our Users Say
              </h2>
              <p className="text-gray-600 max-w-2xl mx-auto">
                See how AI-Guided Learning is transforming education for students and teachers alike.
              </p>
            </div>
            <div className="grid md:grid-cols-3 gap-8">
              {/* Add testimonial cards */}
            </div>
          </div>
        </motion.section>

        {/* Stats Section */}
        <motion.section
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8 }}
          className="py-16 bg-white"
        >
          <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid md:grid-cols-4 gap-8 text-center">
              <div>
                <h3 className="text-4xl font-bold text-secondary mb-2">10k+</h3>
                <p className="text-gray-600">Active Students</p>
              </div>
              <div>
                <h3 className="text-4xl font-bold text-secondary mb-2">500+</h3>
                <p className="text-gray-600">Expert Teachers</p>
              </div>
              <div>
                <h3 className="text-4xl font-bold text-secondary mb-2">95%</h3>
                <p className="text-gray-600">Success Rate</p>
              </div>
              <div>
                <h3 className="text-4xl font-bold text-secondary mb-2">24/7</h3>
                <p className="text-gray-600">AI Support</p>
              </div>
            </div>
          </div>
        </motion.section>

        {/* CTA Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="text-center bg-white/50 backdrop-blur-sm rounded-2xl p-8 md:p-12 border border-gray-100"
        >
          <h2 className="text-2xl md:text-3xl font-bold text-primary mb-4">
            Join us in redefining AI in education
          </h2>
          <p className="text-gray-600 mb-6 max-w-2xl mx-auto">
            No shortcuts—just smarter, more effective learning. Experience the future of education with our AI-guided
            learning platform.
          </p>
          <Button
            size="lg"
            variant="outline"
            className="border-secondary text-secondary hover:bg-secondary hover:text-white transition-colors duration-300"
          >
            Learn More
          </Button>
        </motion.div>
      </div>
    </div>
  );
};

export default Index;
