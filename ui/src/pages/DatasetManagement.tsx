import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { UploadDatasetDialog } from "@/components/UploadDatasetDialog";
import { AppHeader } from "@/components/AppHeader";
import { useToast } from "@/hooks/use-toast";
import { apiClient, formatFileSize } from "@/lib/api-client";
import { useNavigate } from "react-router-dom";
import { 
  Database, 
  FileText, 
  Calendar, 
  HardDrive, 
  Target, 
  Download,
  Trash2,
  Eye,
  RefreshCw,
  ArrowLeft
} from "lucide-react";

interface Dataset {
  dataset_id: string;
  filename: string;
  size_bytes: number;
  created_at: number;
  modified_at: number;
  // Optional fields that might be added later
  name?: string;
  description?: string;
  file_path?: string;
  num_samples?: number;
  num_features?: number;
  target_column?: string;
  data_types?: any;
  preprocessing_config?: any;
}

const DatasetManagement = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const { toast } = useToast();
  const navigate = useNavigate();

  const loadDatasets = async () => {
    try {
      const response = await apiClient.getDatasets();
      
      if (response.error) {
        throw new Error(response.error);
      }
      
      const datasets = response.data?.datasets || [];
      // Map the API response to include name field for display
      const mappedDatasets = datasets.map((dataset: any) => ({
        ...dataset,
        name: dataset.filename, // Use filename as name for display
        file_size: dataset.size_bytes, // Map size_bytes to file_size for compatibility
        created_at: new Date(dataset.created_at * 1000).toISOString() // Convert timestamp to ISO string
      }));
      setDatasets(mappedDatasets);
    } catch (error: any) {
      console.error('Error loading datasets:', error);
      toast({
        title: "Error",
        description: "Failed to load datasets",
        variant: "destructive"
      });
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  const handleRefresh = async () => {
    setRefreshing(true);
    await loadDatasets();
  };

  const handleDownload = async (dataset: Dataset) => {
    try {
      // For now, show a message that download is not implemented
      // In a full implementation, this would download the file from the backend
      toast({
        title: "Download Not Available",
        description: "File download will be implemented in the next version",
        variant: "destructive"
      });
    } catch (error: any) {
      console.error('Error downloading dataset:', error);
      toast({
        title: "Download Failed",
        description: error.message || "Failed to download dataset",
        variant: "destructive"
      });
    }
  };

  const handleDelete = async (dataset: Dataset) => {
    const displayName = dataset.name || dataset.filename;
    if (!confirm(`Are you sure you want to delete "${displayName}"? This action cannot be undone.`)) {
      return;
    }

    try {
      const response = await apiClient.deleteDataset(dataset.dataset_id);

      if (response.error) {
        throw new Error(response.error);
      }

      toast({
        title: "Dataset Deleted",
        description: `${displayName} has been deleted successfully`,
      });

      await loadDatasets();
    } catch (error: any) {
      console.error('Error deleting dataset:', error);
      toast({
        title: "Delete Failed",
        description: error.message || "Failed to delete dataset",
        variant: "destructive"
      });
    }
  };

  useEffect(() => {
    loadDatasets();
  }, []);

  // formatFileSize is now imported from api-client

  const getFileTypeIcon = (config: any) => {
    const extension = config?.extension?.toLowerCase();
    switch (extension) {
      case 'csv':
      case 'tsv':
        return <FileText className="w-5 h-5 text-green-600" />;
      case 'json':
        return <FileText className="w-5 h-5 text-blue-600" />;
      case 'txt':
        return <FileText className="w-5 h-5 text-gray-600" />;
      default:
        return <Database className="w-5 h-5 text-primary" />;
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-background">
        <AppHeader />
        <div className="container mx-auto px-6 py-8">
          <div className="animate-pulse">
            <div className="h-8 bg-muted rounded w-1/4 mb-2"></div>
            <div className="h-4 bg-muted rounded w-1/2 mb-8"></div>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {[1, 2, 3].map(i => (
                <div key={i} className="h-64 bg-muted rounded-lg"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-background">
      <AppHeader />
      
      <div className="container mx-auto px-6 py-8">
        <div className="flex items-center gap-4 mb-8">
          <Button variant="ghost" onClick={() => navigate("/")} size="sm">
            <ArrowLeft className="w-4 h-4" />
            Back to Dashboard
          </Button>
          <div>
            <h1 className="text-3xl font-bold">Dataset Management</h1>
            <p className="text-muted-foreground mt-1">
              Upload, manage, and organize your machine learning datasets
            </p>
          </div>
        </div>

        <div className="flex items-center justify-between mb-8">
          <div>
            <h2 className="text-xl font-semibold">Your Datasets ({datasets.length})</h2>
            <p className="text-muted-foreground text-sm">
              Manage your uploaded datasets and their configurations
            </p>
          </div>
          <div className="flex gap-3">
            <Button 
              variant="outline" 
              size="sm" 
              onClick={handleRefresh}
              disabled={refreshing}
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <UploadDatasetDialog onDatasetUploaded={loadDatasets} />
          </div>
        </div>
        
        {datasets.length === 0 ? (
          <Card className="text-center py-12">
            <CardContent>
              <div className="mx-auto w-24 h-24 bg-primary/10 rounded-full flex items-center justify-center mb-6">
                <Database className="w-12 h-12 text-primary" />
              </div>
              <h3 className="text-xl font-semibold mb-2">No Datasets Yet</h3>
              <p className="text-muted-foreground mb-6">
                Upload your first dataset to get started with AutoML experiments
              </p>
              <UploadDatasetDialog onDatasetUploaded={loadDatasets} />
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {datasets.map((dataset) => (
              <Card key={dataset.dataset_id} className="hover:shadow-lg transition-shadow">
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="flex items-center gap-2">
                      {getFileTypeIcon(dataset.preprocessing_config)}
                      <div>
                        <CardTitle className="text-lg font-semibold line-clamp-1">
                          {dataset.name || dataset.filename}
                        </CardTitle>
                        <CardDescription className="line-clamp-2">
                          {dataset.description || "No description provided"}
                        </CardDescription>
                      </div>
                    </div>
                  </div>
                </CardHeader>
                
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <HardDrive className="w-3 h-3" />
                        Size
                      </div>
                      <div className="font-medium">{formatFileSize(dataset.file_size || dataset.size_bytes)}</div>
                    </div>
                    
                    <div>
                      <div className="flex items-center gap-1 text-muted-foreground">
                        <FileText className="w-3 h-3" />
                        Type
                      </div>
                      <div className="font-medium">
                        {dataset.preprocessing_config?.extension?.toUpperCase() || 'Unknown'}
                      </div>
                    </div>
                    
                    {dataset.num_samples && (
                      <div>
                        <div className="flex items-center gap-1 text-muted-foreground">
                          <Database className="w-3 h-3" />
                          Samples
                        </div>
                        <div className="font-medium">{dataset.num_samples.toLocaleString()}</div>
                      </div>
                    )}
                    
                    {dataset.num_features && (
                      <div>
                        <div className="flex items-center gap-1 text-muted-foreground">
                          <Target className="w-3 h-3" />
                          Features
                        </div>
                        <div className="font-medium">{dataset.num_features}</div>
                      </div>
                    )}
                  </div>
                  
                  {dataset.target_column && (
                    <div className="text-sm">
                      <div className="text-muted-foreground">Target Column</div>
                      <Badge variant="secondary" className="text-xs">
                        {dataset.target_column}
                      </Badge>
                    </div>
                  )}
                  
                  <div className="text-xs text-muted-foreground flex items-center gap-1">
                    <Calendar className="w-3 h-3" />
                    Uploaded {new Date(dataset.created_at).toLocaleDateString()}
                  </div>
                  
                  <div className="flex gap-2 pt-2">
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleDownload(dataset)}
                      className="flex-1"
                    >
                      <Download className="w-3 h-3" />
                      Download
                    </Button>
                    <Button 
                      size="sm" 
                      variant="outline"
                      onClick={() => handleDelete(dataset)}
                      className="text-destructive hover:text-destructive"
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetManagement;