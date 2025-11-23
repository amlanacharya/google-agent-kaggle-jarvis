"""
IoT Controller Agent for smart home and device management.

Handles:
- Device discovery and control
- Home automation scenes
- Energy monitoring
- Security system integration
- Environmental monitoring
"""

from typing import Dict, List, Optional, Any
import asyncio
import json

from .base_agent import BaseAgent
from ..core.logger import logger


class IoTControllerAgent(BaseAgent):
    """Agent specialized in IoT device management."""

    def __init__(self, memory=None, llm=None):
        """Initialize IoT controller agent."""
        super().__init__(
            name="iot_controller",
            capabilities=[
                "discover_devices",
                "control_device",
                "create_scene",
                "monitor_energy",
                "set_automation",
                "check_security",
                "environmental_status",
                "device_grouping"
            ],
            memory=memory,
            llm=llm
        )
        self.devices = {}
        self.scenes = {}
        self.automations = {}
        self._initialize_devices()

    def _initialize_devices(self):
        """Initialize device registry."""
        # Mock device registry for demo
        # In production, this would connect to actual IoT platforms
        self.devices = {
            "living_room_light": {
                "id": "light_001",
                "type": "smart_bulb",
                "location": "living_room",
                "state": "off",
                "brightness": 0,
                "color": None,
                "protocol": "zigbee"
            },
            "bedroom_thermostat": {
                "id": "thermo_001",
                "type": "thermostat",
                "location": "bedroom",
                "state": "auto",
                "temperature": 22,
                "target_temp": 22,
                "mode": "cool",
                "protocol": "wifi"
            },
            "front_door_lock": {
                "id": "lock_001",
                "type": "smart_lock",
                "location": "front_door",
                "state": "locked",
                "protocol": "zwave"
            },
            "garage_door": {
                "id": "garage_001",
                "type": "garage_opener",
                "location": "garage",
                "state": "closed",
                "protocol": "wifi"
            },
            "security_camera_front": {
                "id": "cam_001",
                "type": "camera",
                "location": "front_yard",
                "state": "active",
                "recording": True,
                "protocol": "wifi"
            },
            "kitchen_speaker": {
                "id": "speaker_001",
                "type": "smart_speaker",
                "location": "kitchen",
                "state": "idle",
                "volume": 50,
                "protocol": "wifi"
            }
        }

        logger.info(f"Initialized {len(self.devices)} IoT devices")

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process IoT control task."""
        action = task.get("action", "")

        try:
            if action == "discover":
                return await self._discover_devices(task)
            elif action == "control":
                return await self._control_device(task)
            elif action == "create_scene":
                return await self._create_scene(task)
            elif action == "activate_scene":
                return await self._activate_scene(task)
            elif action == "monitor_energy":
                return await self._monitor_energy(task)
            elif action == "set_automation":
                return await self._set_automation(task)
            elif action == "security_status":
                return await self._check_security(task)
            elif action == "environmental":
                return await self._environmental_status(task)
            elif action == "smart_control":
                return await self._smart_control(task)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
        except Exception as e:
            logger.error(f"IoT controller error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def _discover_devices(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Discover available IoT devices."""
        device_type = task.get('type', None)
        location = task.get('location', None)

        filtered_devices = self.devices

        if device_type:
            filtered_devices = {
                k: v for k, v in filtered_devices.items()
                if v['type'] == device_type
            }

        if location:
            filtered_devices = {
                k: v for k, v in filtered_devices.items()
                if v['location'] == location
            }

        return {
            "success": True,
            "devices": filtered_devices,
            "count": len(filtered_devices)
        }

    async def _control_device(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Control a specific device."""
        device_name = task.get('device')
        command = task.get('command')
        params = task.get('params', {})

        if device_name not in self.devices:
            return {
                "success": False,
                "error": f"Device '{device_name}' not found"
            }

        device = self.devices[device_name]

        # Simulate device control
        if command == "turn_on":
            device['state'] = "on"
            if 'brightness' in device:
                device['brightness'] = params.get('brightness', 100)
        elif command == "turn_off":
            device['state'] = "off"
            if 'brightness' in device:
                device['brightness'] = 0
        elif command == "set_temperature":
            if device['type'] == "thermostat":
                device['target_temp'] = params.get('temperature', 22)
        elif command == "lock":
            if device['type'] == "smart_lock":
                device['state'] = "locked"
        elif command == "unlock":
            if device['type'] == "smart_lock":
                device['state'] = "unlocked"
        elif command == "set_brightness":
            if 'brightness' in device:
                device['brightness'] = params.get('level', 50)
        elif command == "set_color":
            if 'color' in device:
                device['color'] = params.get('color', 'white')

        logger.info(f"Controlled device {device_name}: {command}")

        return {
            "success": True,
            "device": device_name,
            "command": command,
            "new_state": device
        }

    async def _create_scene(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Create a scene with multiple device states."""
        scene_name = task.get('name')
        device_states = task.get('devices', {})

        self.scenes[scene_name] = {
            'name': scene_name,
            'devices': device_states,
            'created_at': 'now'
        }

        logger.info(f"Created scene: {scene_name}")

        return {
            "success": True,
            "scene": scene_name,
            "devices_configured": len(device_states)
        }

    async def _activate_scene(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Activate a predefined scene."""
        scene_name = task.get('scene')

        if scene_name not in self.scenes:
            return {
                "success": False,
                "error": f"Scene '{scene_name}' not found"
            }

        scene = self.scenes[scene_name]
        results = []

        # Apply all device states in the scene
        for device_name, state in scene['devices'].items():
            result = await self._control_device({
                'device': device_name,
                'command': state.get('command'),
                'params': state.get('params', {})
            })
            results.append(result)

        return {
            "success": True,
            "scene": scene_name,
            "devices_controlled": len(results),
            "results": results
        }

    async def _monitor_energy(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor energy consumption."""
        # Mock energy data
        energy_data = {
            "total_consumption_kwh": 45.7,
            "current_usage_watts": 2340,
            "devices": {
                "hvac": {"usage_watts": 1500, "percentage": 64.1},
                "lighting": {"usage_watts": 350, "percentage": 15.0},
                "appliances": {"usage_watts": 490, "percentage": 20.9}
            },
            "cost_today": 5.48,
            "projected_monthly": 164.40,
            "recommendations": []
        }

        # Use AI to generate recommendations
        prompt = f"""
        Analyze this energy consumption data and provide recommendations:
        {json.dumps(energy_data, indent=2)}

        Suggest:
        1. Ways to reduce consumption
        2. Optimal device scheduling
        3. Energy-saving automations
        """

        recommendations = await self.llm.generate(prompt)
        energy_data['recommendations'] = recommendations

        return {
            "success": True,
            "energy_data": energy_data
        }

    async def _set_automation(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Set up device automation."""
        automation_name = task.get('name')
        trigger = task.get('trigger')
        actions = task.get('actions')

        self.automations[automation_name] = {
            'name': automation_name,
            'trigger': trigger,
            'actions': actions,
            'enabled': True
        }

        logger.info(f"Created automation: {automation_name}")

        return {
            "success": True,
            "automation": automation_name,
            "trigger": trigger,
            "actions_count": len(actions)
        }

    async def _check_security(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Check security system status."""
        security_devices = {
            k: v for k, v in self.devices.items()
            if v['type'] in ['smart_lock', 'camera', 'sensor']
        }

        security_status = {
            "armed": True,
            "devices": security_devices,
            "alerts": [],
            "recent_events": []
        }

        # Check for issues
        for name, device in security_devices.items():
            if device['type'] == 'smart_lock' and device['state'] == 'unlocked':
                security_status['alerts'].append(f"Alert: {name} is unlocked")

        return {
            "success": True,
            "security_status": security_status,
            "devices_monitored": len(security_devices),
            "alert_count": len(security_status['alerts'])
        }

    async def _environmental_status(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Get environmental sensor status."""
        # Mock environmental data
        environmental_data = {
            "temperature": {
                "living_room": 23.5,
                "bedroom": 22.0,
                "outside": 18.3
            },
            "humidity": {
                "living_room": 45,
                "bedroom": 48,
                "outside": 62
            },
            "air_quality": {
                "pm25": 12,
                "co2": 450,
                "voc": 0.5
            },
            "comfort_index": 8.5
        }

        # Use AI for recommendations
        prompt = f"""
        Analyze environmental conditions and provide comfort recommendations:
        {json.dumps(environmental_data, indent=2)}

        Consider:
        - Optimal temperature and humidity
        - Air quality concerns
        - Seasonal adjustments
        """

        recommendations = await self.llm.generate(prompt)

        return {
            "success": True,
            "environmental_data": environmental_data,
            "recommendations": recommendations
        }

    async def _smart_control(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Use AI for intelligent device control."""
        user_request = task.get('request')

        # Use LLM to understand intent and generate control plan
        prompt = f"""
        User request: "{user_request}"

        Available devices:
        {json.dumps(self.devices, indent=2)}

        Determine:
        1. Which devices to control
        2. What commands to execute
        3. Optimal sequence and timing

        Return JSON with action plan.
        """

        control_plan = await self.llm.generate(prompt)

        # Execute the plan (simplified for demo)
        return {
            "success": True,
            "user_request": user_request,
            "control_plan": control_plan,
            "message": "Smart control plan generated"
        }
