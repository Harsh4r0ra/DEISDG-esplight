import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Activity, Power } from 'lucide-react';

const GridDashboard = () => {
  const [gridState, setGridState] = useState({
    0: false,
    1: false,
    2: false,
    3: false
  });

  // Simulate receiving updates from the backend
  useEffect(() => {
    const socket = new WebSocket('ws://your-backend-url/ws');

    socket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setGridState(prevState => ({
        ...prevState,
        [data.gridIndex]: data.isActive
      }));
    };

    return () => {
      socket.close();
    };
  }, []);

  const renderGrid = () => {
    return (
      <div className="grid grid-cols-2 gap-4 p-4">
        {Object.entries(gridState).map(([index, isActive]) => (
          <Card key={index} className={`p-6 ${isActive ? 'bg-green-100' : 'bg-red-50'}`}>
            <CardHeader className="flex flex-row items-center justify-between p-2">
              <CardTitle className="text-lg font-semibold">
                Zone {parseInt(index) + 1}
              </CardTitle>
              {isActive ? (
                <Activity className="h-6 w-6 text-green-600" />
              ) : (
                <Power className="h-6 w-6 text-red-400" />
              )}
            </CardHeader>
            <CardContent>
              <div className="mt-2 text-sm">
                <p>Status: {isActive ? 'Active' : 'Inactive'}</p>
                <p className="mt-1">ESP8266-{parseInt(index) + 1}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  };

  return (
    <div className="max-w-4xl mx-auto p-6">
      <h1 className="text-2xl font-bold mb-6">Room Activity Monitor</h1>
      {renderGrid()}
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h2 className="text-lg font-semibold mb-2">System Status</h2>
        <div className="flex space-x-4">
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-green-500 mr-2"></div>
            <span>Camera Connected</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 rounded-full bg-blue-500 mr-2"></div>
            <span>Motion Detection Active</span>
          </div>
        </div>
      </div>
      <div className="mt-6 p-4 bg-gray-50 rounded-lg">
        <h2 className="text-lg font-semibold mb-2">Live Video Feed</h2>
        <img src="http://10.9.84.177:4747/video" alt="http://10.9.84.177:4747/video" width="425" height="319" class="shrinkToFit"/>
      </div>

    </div>
  );
};

export default GridDashboard;