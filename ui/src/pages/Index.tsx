import { AppHeader } from "@/components/AppHeader";
import { AutoMLHeader } from "@/components/AutoMLHeader";
import { ProjectDashboard } from "@/components/ProjectDashboard";
import { AutoMLWorkflow } from "@/components/AutoMLWorkflow";

const Index = () => {
  return (
    <div className="min-h-screen bg-gradient-background">
      <AppHeader />
      <AutoMLHeader />
      <ProjectDashboard />
      <AutoMLWorkflow />
    </div>
  );
};

export default Index;
