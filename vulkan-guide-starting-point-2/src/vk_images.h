
#pragma once 

#include <vulkan/vulkan.h>

namespace vkutil {

	void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
	void transition_image(VkCommandBuffer cmd, VkImage image, uint32_t baseArrayLayer,VkImageLayout currentLayout, VkImageLayout newLayout);
	void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout, VkImageAspectFlags aspectFlags);
	void transition_image(VkCommandBuffer cmd, VkImage image, uint32_t baseArrayLayer, VkImageLayout currentLayout, VkImageLayout newLayout, VkImageAspectFlags aspectFlags);
	void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize);

	void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize);
};