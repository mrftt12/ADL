import { useState, useEffect } from "react";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useAuth } from "@/contexts/AuthContext";
import { apiClient } from "@/lib/api-client";
import { Play, Loader2, AlertCircle } from "lucide-react";

interface CreateExperimentDialogProps {
  onExperimentCreated?: () => void;
}

export const CreateExperimentDialog = ({ onExperimentCreated }: CreateExperimentDialogProps) => {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<any>(null);
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    dataset_id: "",
    task_type: "classification",
    data_type: "tabular",
    target_column: ""
  });
  const { toast } = useToast();
  const { user, isAuthenticated } = useAuth();

  useEffect(() => {
    if (open) {
      loadDatasets();
    }
  }, [open]);

  const loadDatasets = async () => {
    if (!isAuthenticated) {
      toast({
        title: "Authentication Required",
        description: "Please sign in to view your datasets",
        variant: "destructive"
      });
      return;
    }

    setLoadingDatasets(true);
    try {
      const response = await apiClient.getDatasets();
      if (response.error) {
        throw new Error(response.error);
      }
      if (response.data) {
        setDatasets(response.data.datasets);
      }
    } catch (error: any) {
      console.error('Error loading datasets:', error);
      toast({
        title: "Error Loading Datasets",
        description: error.message || "Failed to load datasets. Please try again.",
        variant: "destructive"
      });
    } finally {
      setLoadingDatasets(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!isAuthenticated) {
      toast({
        title: "Authentication Required",
        description: "Please sign in to create experiments",
        variant: "destructive"
      });
      return;
    }
    
    if (!formData.name || !formData.dataset_id) {
      toast({
        title: "Validation Error",
        description: "Please enter an experiment name and select a dataset",
        variant: "destructive"
      });
      return;
    }

    setLoading(true);

    try {
      // Get the selected dataset info
      const dataset = datasets.find(d => d.dataset_id === formData.dataset_id);
      const dataset_path = dataset ? `data/uploads/${dataset.filename}` : formData.dataset_id;

      // Create experiment
      const response = await apiClient.createExperiment({
        name: formData.name,
        dataset_path: dataset_path,
        task_type: formData.task_type,
        data_type: formData.data_type,
        target_column: formData.target_column || undefined,
        config: {
          description: formData.description,
          dataset_id: formData.dataset_id,
          user_id: user?.id
        }
      });

      if (response.error) {
        if (response.error.includes('authentication') || response.error.includes('unauthorized')) {
          throw new Error("Authentication failed. Please sign in again.");
        }
        throw new Error(response.error);
      }

      const experiment = response.data!;

      toast({
        title: "Experiment Created",
        description: `${formData.name} has been created successfully`,
      });

      // Optionally start the experiment immediately
      try {
        const startResponse = await apiClient.startExperiment(experiment.id);
        if (startResponse.error) {
          console.warn('Failed to start experiment:', startResponse.error);
          toast({
            title: "Experiment Created",
            description: `${formData.name} was created but failed to start automatically. You can start it manually.`,
            variant: "default"
          });
        } else {
          toast({
            title: "Experiment Started",
            description: `${formData.name} is now running`,
          });
        }
      } catch (startError) {
        console.warn('Failed to start experiment:', startError);
        // Don't fail if we can't start immediately
      }

      setFormData({ 
        name: "", 
        description: "",
        dataset_id: "",
        task_type: "classification",
        data_type: "tabular",
        target_column: ""
      });
      setSelectedDataset(null);
      setOpen(false);
      onExperimentCreated?.();
      
    } catch (error: any) {
      console.error('Error creating experiment:', error);
      
      let errorMessage = "Failed to create experiment";
      if (error.message) {
        errorMessage = error.message;
      }
      
      // Handle specific authentication errors
      if (errorMessage.includes('authentication') || errorMessage.includes('unauthorized') || errorMessage.includes('401')) {
        errorMessage = "Authentication failed. Please sign in again.";
      }
      
      toast({
        title: "Error Creating Experiment",
        description: errorMessage,
        variant: "destructive"
      });
    } finally {
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFormData({ 
      name: "", 
      description: "",
      dataset_id: "",
      task_type: "classification",
      data_type: "tabular",
      target_column: ""
    });
    setSelectedDataset(null);
  };

  const handleDatasetChange = (datasetId: string) => {
    const dataset = datasets.find(d => d.dataset_id === datasetId);
    setSelectedDataset(dataset);
    setFormData(prev => ({ ...prev, dataset_id: datasetId }));
  };

  return (
    <Dialog open={open} onOpenChange={(newOpen) => {
      setOpen(newOpen);
      if (!newOpen) resetForm();
    }}>
      <DialogTrigger asChild>
        <Button variant="gradient">
          <Play className="w-4 h-4" />
          New Experiment
        </Button>
      </DialogTrigger>
      
      <DialogContent className="sm:max-w-[500px]">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold">Create New Experiment</DialogTitle>
        </DialogHeader>
        
        {!isAuthenticated && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <div className="flex items-center">
              <AlertCircle className="w-5 h-5 text-yellow-600 mr-2" />
              <p className="text-sm text-yellow-800">
                You need to be signed in to create experiments. Please sign in first.
              </p>
            </div>
          </div>
        )}
        
        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Experiment Name *</Label>
              <Input
                id="name"
                placeholder="e.g., Customer Classification Model"
                value={formData.name}
                onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                disabled={loading}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="dataset">Dataset *</Label>
              {loadingDatasets ? (
                <div className="flex items-center justify-center p-4 border rounded-md">
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                  <span className="text-sm text-muted-foreground">Loading datasets...</span>
                </div>
              ) : datasets.length === 0 ? (
                <div className="p-4 border rounded-md bg-gray-50">
                  <p className="text-sm text-muted-foreground text-center">
                    No datasets found. Please upload a dataset first.
                  </p>
                </div>
              ) : (
                <Select 
                  value={formData.dataset_id} 
                  onValueChange={handleDatasetChange}
                  disabled={loading || !isAuthenticated}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select a dataset" />
                  </SelectTrigger>
                  <SelectContent>
                    {datasets.map((dataset) => (
                      <SelectItem key={dataset.dataset_id} value={dataset.dataset_id}>
                        <div className="flex flex-col">
                          <span className="font-medium">{dataset.filename}</span>
                          <span className="text-xs text-muted-foreground">
                            {(dataset.size_bytes / 1024 / 1024).toFixed(2)} MB
                          </span>
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              {selectedDataset && (
                <div className="text-xs text-muted-foreground mt-1">
                  Selected: {selectedDataset.filename} ({(selectedDataset.size_bytes / 1024 / 1024).toFixed(2)} MB)
                </div>
              )}
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="task_type">Task Type</Label>
                <Select 
                  value={formData.task_type} 
                  onValueChange={(value) => setFormData(prev => ({ ...prev, task_type: value }))}
                  disabled={loading}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="classification">Classification</SelectItem>
                    <SelectItem value="regression">Regression</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label htmlFor="data_type">Data Type</Label>
                <Select 
                  value={formData.data_type} 
                  onValueChange={(value) => setFormData(prev => ({ ...prev, data_type: value }))}
                  disabled={loading}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="tabular">Tabular</SelectItem>
                    <SelectItem value="text">Text</SelectItem>
                    <SelectItem value="image">Image</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="target_column">Target Column (Optional)</Label>
              <Input
                id="target_column"
                placeholder="e.g., label, target, y"
                value={formData.target_column}
                onChange={(e) => setFormData(prev => ({ ...prev, target_column: e.target.value }))}
                disabled={loading}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Description (Optional)</Label>
              <Textarea
                id="description"
                placeholder="Describe the experiment goals or approach..."
                value={formData.description}
                onChange={(e) => setFormData(prev => ({ ...prev, description: e.target.value }))}
                disabled={loading}
                rows={3}
              />
            </div>
          </div>

          <div className="flex justify-end space-x-3 pt-4">
            <Button
              type="button"
              variant="outline"
              onClick={() => setOpen(false)}
              disabled={loading}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              variant="gradient"
              disabled={loading || !isAuthenticated || datasets.length === 0}
            >
              {loading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Creating...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4" />
                  Start Experiment
                </>
              )}
            </Button>
          </div>
        </form>
      </DialogContent>
    </Dialog>
  );
};