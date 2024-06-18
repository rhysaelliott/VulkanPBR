// vulkan_guide.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <vk_types.h>



constexpr unsigned int FRAME_OVERLAP = 2;

class VulkanEngine {
public:
	DelectionQueue _mainDeletionQueue;

	bool _isInitialized{ false };
	int _frameNumber {0};
	bool stop_rendering{ false };
	VkExtent2D _windowExtent{ 1700 , 900 };

	struct SDL_Window* _window{ nullptr };

	static VulkanEngine& Get();

	//initialisation members
	VkInstance _instance;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkPhysicalDevice _chosenGPU;
	VkDevice _device;
	VkSurfaceKHR _surface;

	//swapchain members
	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;
	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	//draw resources
	AllocatedImage _drawImage;
	VkExtent2D _drawExtent;

	//frame members
	FrameData _frames[FRAME_OVERLAP];

	//graphics members
	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;

	//memory allocator
	VmaAllocator _allocator;

	//initializes everything in the engine
	void init();

	//shuts down the engine
	void cleanup();

	//draw loop
	void draw();

	void draw_background(VkCommandBuffer cmd);

	//run main loop
	void run();

	//frame member functions
	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

private:
	void init_vulkan();
	void init_swapchain();
	void init_commands();
	void init_sync_structures();

	void create_swapchain(uint32_t width, uint32_t height);
	void destroy_swapchain();
};
